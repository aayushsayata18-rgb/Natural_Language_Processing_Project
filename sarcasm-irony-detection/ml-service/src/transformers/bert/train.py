"""
Phase 8 — BERT Fine-Tuning Engine
=================================
Sarcasm & Irony Detection System using NLP

Features:
- AdamW optimizer with weight decay (0.01)
- Linear learning-rate schedule with warmup (warmup_ratio=0.1)
- Multiclass CrossEntropyLoss on raw logits (optional class weights support)
- Gradient clipping (max_norm=1.0)
- Validation tracking per epoch (loss, accuracy, Macro F1)
- Checkpoint selection strictly based on validation Macro F1
- Early stopping with patience
- Training history logging
"""

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.transformers.bert.evaluate import evaluate_bert_dataloader

logger = logging.getLogger(__name__)


def train_bert_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    patience: int = 2,
    class_weights: Optional[torch.Tensor] = None,
    save_best_to_memory: bool = True,
    max_train_steps_per_epoch: Optional[int] = None,
    max_val_steps_per_epoch: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Executes fine-tuning of BERT on train_loader with validation monitoring.
    """
    model.to(device)

    # Apply class weights to loss if provided
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        logger.info(f"Using weighted CrossEntropyLoss: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Prepare optimizer parameters with weight decay exclusion for LayerNorm / bias
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    steps_per_epoch = min(len(train_loader), max_train_steps_per_epoch or len(train_loader))
    total_training_steps = steps_per_epoch * epochs
    warmup_steps = int(total_training_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    logger.info(
        f"Starting BERT fine-tuning: epochs={epochs}, lr={learning_rate}, "
        f"weight_decay={weight_decay}, total_steps={total_training_steps}, warmup_steps={warmup_steps}"
    )

    history: List[Dict[str, Any]] = []
    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_model_weights = None
    no_improvement_count = 0
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        step_count = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            if max_train_steps_per_epoch and batch_idx > max_train_steps_per_epoch:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits

            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            train_loss_total += loss.item() * len(labels)
            train_samples += len(labels)
            step_count += 1

            if batch_idx % 100 == 0 or batch_idx == steps_per_epoch:
                current_avg_loss = train_loss_total / train_samples
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch}/{epochs} [{batch_idx}/{steps_per_epoch}] "
                    f"Train Loss: {current_avg_loss:.4f} | LR: {current_lr:.2e}"
                )

        avg_train_loss = train_loss_total / train_samples if train_samples > 0 else 0.0

        # Validation phase
        val_metrics, _, _, _ = evaluate_bert_dataloader(
            model=model,
            dataloader=val_loader,
            device=device,
            criterion=criterion,
            max_steps=max_val_steps_per_epoch,
        )

        val_loss = val_metrics.get("loss", 0.0)
        val_acc = val_metrics.get("accuracy", 0.0)
        val_macro_f1 = val_metrics.get("macro_f1", 0.0)
        epoch_duration = time.time() - epoch_start_time

        logger.info(
            f"--- Epoch {epoch}/{epochs} Summary --- "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val Macro F1: {val_macro_f1:.4f} | Duration: {epoch_duration:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": round(float(avg_train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "val_accuracy": round(float(val_acc), 4),
            "val_macro_f1": round(float(val_macro_f1), 4),
            "val_weighted_f1": round(float(val_metrics.get("weighted_f1", 0.0)), 4),
            "epoch_duration_s": round(float(epoch_duration), 2),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)

        # Checkpoint selection strictly on validation Macro F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            no_improvement_count = 0
            if save_best_to_memory:
                best_model_weights = copy.deepcopy(model.state_dict())
            logger.info(f"==> New best model checkpoint at Epoch {epoch} with Val Macro F1: {val_macro_f1:.4f}")
        else:
            no_improvement_count += 1
            logger.info(f"No improvement in Val Macro F1 for {no_improvement_count} epoch(s).")
            if no_improvement_count >= patience:
                logger.info(f"Early stopping triggered at Epoch {epoch} (patience={patience}).")
                break

    total_training_time = time.time() - start_total_time

    # Restore best checkpoint
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        logger.info(f"Restored best model weights from Epoch {best_epoch} (Val Macro F1: {best_val_macro_f1:.4f})")

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(float(best_val_macro_f1), 4),
        "total_epochs_trained": len(history),
        "total_training_time_s": round(float(total_training_time), 2),
        "history": history,
    }

    return model, summary
