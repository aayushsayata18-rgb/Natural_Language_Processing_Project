"""
PyTorch training loop, dataloaders, learning rate scheduling, and early stopping for Bi-LSTM.
"""

import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def create_dataloader(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Constructs a PyTorch DataLoader from feature matrices and label vectors.
    """
    if isinstance(X, np.ndarray):
        tensor_x = torch.tensor(X, dtype=torch.long)
    else:
        tensor_x = X.long()

    if isinstance(y, np.ndarray):
        tensor_y = torch.tensor(y, dtype=torch.long)
    else:
        tensor_y = y.long()

    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 30,
    learning_rate: float = 0.001,
    patience: int = 5,
    lr_patience: int = 2,
    lr_factor: float = 0.5,
    checkpoint_dir: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Trains model with early stopping, ReduceLROnPlateau, and checkpointing.
    """
    device_obj = torch.device(device)
    model.to(device_obj)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
    )
    early_stopping = EarlyStopping(patience=patience)

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device_obj)
            batch_y = batch_y.to(device_obj)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            total_train += len(batch_y)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device_obj)
                batch_y = batch_y.to(device_obj)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * len(batch_y)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                total_val += len(batch_y)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_correct / total_val
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(current_lr)

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            if checkpoint_dir:
                torch.save(best_weights, Path(checkpoint_dir) / "best_model.pth")
                torch.save(best_weights, Path(checkpoint_dir) / "best_bilstm.pt")

        print(
            f"Epoch {epoch:2d}/{max_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if early_stopping(epoch_val_loss, epoch):
            print(f"Early stopping triggered at epoch {epoch}. Best epoch: {early_stopping.best_epoch}")
            break

    # Restore best weights
    model.load_state_dict(best_weights)
    return model, history
