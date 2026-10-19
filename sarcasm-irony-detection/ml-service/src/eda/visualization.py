"""
visualization.py — All EDA visualizations.

Creates publication-quality charts with consistent styling.
Every visualization has: title, axis labels, readable font, appropriate scale, legend.

Phase 2: Exploratory Data Analysis
Project: Sarcasm & Irony Detection System
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, List, Tuple, Optional

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


# ---------------------------------------------------------------------------
# Global Style Configuration
# ---------------------------------------------------------------------------

CLASS_ORDER = ["Regular", "Sarcasm", "Irony", "Figurative"]
CLASS_COLORS = {
    "Regular": "#4CAF50",
    "Sarcasm": "#FF5722",
    "Irony": "#2196F3",
    "Figurative": "#9C27B0",
}
COLOR_PALETTE = [CLASS_COLORS[c] for c in CLASS_ORDER]

def setup_style():
    """Configure consistent matplotlib/seaborn style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def _save_fig(fig, path: str):
    """Save figure and close."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 1. Class Distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    dist_df: pd.DataFrame,
    title: str,
    save_path: str,
):
    """Bar chart of class distribution with counts and percentages."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        dist_df["Class"].astype(str),
        dist_df["Count"],
        color=COLOR_PALETTE[:len(dist_df)],
        edgecolor="white",
        linewidth=1.5,
    )

    # Add count labels on bars
    for bar, pct in zip(bars, dist_df["Percentage"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(dist_df["Count"]) * 0.01,
            f"{int(height):,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel("Number of Tweets", fontsize=13)
    ax.set_ylim(0, max(dist_df["Count"]) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Histograms
# ---------------------------------------------------------------------------

def plot_histogram(
    features: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    save_path: str,
    bins: int = 50,
):
    """Histogram of a numeric feature."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        features[column], bins=bins, color="#2196F3",
        edgecolor="white", alpha=0.85,
    )
    ax.axvline(
        features[column].mean(), color="#FF5722",
        linestyle="--", linewidth=2, label=f"Mean: {features[column].mean():.1f}",
    )
    ax.axvline(
        features[column].median(), color="#4CAF50",
        linestyle="-.", linewidth=2, label=f"Median: {features[column].median():.1f}",
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Boxplots by Class
# ---------------------------------------------------------------------------

def plot_boxplot_by_class(
    features: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    save_path: str,
):
    """Boxplot of a numeric feature grouped by class."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=features, x="label", y=column,
        order=CLASS_ORDER, palette=CLASS_COLORS,
        ax=ax, fliersize=2, linewidth=1.2,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Top N-gram Bar Charts
# ---------------------------------------------------------------------------

def plot_top_ngrams(
    ngrams: List[Tuple[str, int]],
    title: str,
    save_path: str,
    color: str = "#2196F3",
    top_n: int = 20,
):
    """Horizontal bar chart for top n-grams."""
    setup_style()
    if not ngrams:
        return

    ngrams = ngrams[:top_n]
    words = [ng for ng, _ in ngrams]
    counts = [c for _, c in ngrams]

    fig, ax = plt.subplots(figsize=(10, max(6, len(words) * 0.35)))

    bars = ax.barh(
        range(len(words)), counts, color=color,
        edgecolor="white", alpha=0.85,
    )
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}", va="center", fontsize=9,
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Frequency", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    sns.despine()

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Social-Media Feature by Class
# ---------------------------------------------------------------------------

def plot_feature_by_class(
    data: List[Dict],
    title: str,
    ylabel: str,
    save_path: str,
    use_percentage: bool = True,
):
    """Bar chart of a feature metric by class."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    classes = [d["class"] for d in data]
    values = [d.get("percentage", d.get("mean", 0)) for d in data]
    colors = [CLASS_COLORS.get(c, "#999") for c in classes]

    bars = ax.bar(classes, values, color=colors, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        fmt = f"{val:.1f}%" if use_percentage else f"{val:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            fmt, ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Punctuation Multi-Feature by Class
# ---------------------------------------------------------------------------

def plot_punctuation_by_class(
    social_features: pd.DataFrame,
    save_path: str,
):
    """Grouped bar chart showing punctuation features by class."""
    setup_style()

    punct_features = ["exclamation_count", "question_count", "ellipsis_count"]
    labels = ["Exclamation (!)", "Question (?)", "Ellipsis (...)"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(CLASS_ORDER))
    width = 0.25
    feature_colors = ["#FF5722", "#2196F3", "#9C27B0"]

    for i, (feat, label, color) in enumerate(zip(punct_features, labels, feature_colors)):
        means = []
        for cls in CLASS_ORDER:
            subset = social_features[social_features["label"] == cls]
            means.append((subset[feat] > 0).mean() * 100)
        ax.bar(x + i * width, means, width, label=label, color=color, alpha=0.85)

    ax.set_title(
        "Punctuation Pattern Presence by Class (%)",
        fontsize=16, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel("Tweets with Feature (%)", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_ORDER)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 7. Word Clouds
# ---------------------------------------------------------------------------

def plot_wordcloud(
    text: str,
    title: str,
    save_path: str,
    colormap: str = "viridis",
):
    """Generate and save a word cloud."""
    if not HAS_WORDCLOUD:
        print(f"  Skipped word cloud (wordcloud package not installed): {title}")
        return

    setup_style()
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap=colormap,
        max_words=100,
        contour_width=1,
        contour_color="steelblue",
    )

    try:
        wc.generate(text)
    except ValueError:
        print(f"  Skipped word cloud (no words): {title}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.axis("off")

    _save_fig(fig, save_path)


def generate_all_wordclouds(
    df: pd.DataFrame,
    text_col: str,
    output_dir: str,
):
    """Generate word clouds for overall and each class."""
    colormaps = {
        "Overall": "viridis",
        "Regular": "Greens",
        "Sarcasm": "Oranges",
        "Irony": "Blues",
        "Figurative": "Purples",
    }

    # Overall
    all_text = " ".join(df[text_col].dropna())
    plot_wordcloud(
        all_text,
        "Word Cloud — Overall",
        os.path.join(output_dir, "wordcloud_overall.png"),
        colormap=colormaps["Overall"],
    )

    # Per class
    for label in CLASS_ORDER:
        subset = df[df["label"] == label]
        class_text = " ".join(subset[text_col].dropna())
        plot_wordcloud(
            class_text,
            f"Word Cloud — {label}",
            os.path.join(output_dir, f"wordcloud_{label.lower()}.png"),
            colormap=colormaps.get(label, "viridis"),
        )


# ---------------------------------------------------------------------------
# 8. Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    features: pd.DataFrame,
    columns: List[str],
    save_path: str,
):
    """Correlation heatmap for numeric EDA features."""
    setup_style()

    # Filter to existing columns
    cols = [c for c in columns if c in features.columns]
    if len(cols) < 2:
        print("  Skipped correlation heatmap (not enough numeric columns)")
        return

    corr = features[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        "Feature Correlation Heatmap (EDA Features)",
        fontsize=16, fontweight="bold", pad=15,
    )
    # Pretty labels
    label_map = {
        "character_count": "Char Count",
        "word_count": "Word Count",
        "punctuation_count": "Punct Count",
        "hashtag_count": "Hashtag Count",
        "mention_count": "Mention Count",
        "emoji_count": "Emoji Count",
        "uppercase_ratio": "Uppercase Ratio",
        "exclamation_count": "Exclamation",
        "question_count": "Question",
        "url_count": "URL Count",
    }
    xlabels = [label_map.get(c, c) for c in cols]
    ylabels = [label_map.get(c, c) for c in cols]
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels, rotation=0)

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 9. Feature Distribution Violin/Boxplots
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    features: pd.DataFrame,
    columns: List[str],
    save_path: str,
):
    """
    Multi-panel boxplot showing several feature distributions by class.
    """
    setup_style()
    cols = [c for c in columns if c in features.columns]
    n_cols = len(cols)
    if n_cols == 0:
        return

    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, min(3, n_cols), figsize=(16, 5 * n_rows))
    if n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(cols):
        if i >= len(axes):
            break
        label_map = {
            "character_count": "Character Count",
            "word_count": "Word Count",
            "punctuation_count": "Punctuation Count",
            "hashtag_count": "Hashtag Count",
            "mention_count": "Mention Count",
            "emoji_count": "Emoji Count",
            "uppercase_ratio": "Uppercase Ratio",
        }
        sns.boxplot(
            data=features, x="label", y=col,
            order=CLASS_ORDER, palette=CLASS_COLORS,
            ax=axes[i], fliersize=1.5,
        )
        axes[i].set_title(label_map.get(col, col), fontsize=13, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].grid(axis="y", alpha=0.3)

    # Hide empty axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Feature Distributions by Class",
        fontsize=18, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, save_path)
