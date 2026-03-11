"""Visualization manager using Matplotlib.

Generates all chart types needed for OPM Repeatability analysis:
- Profile Overlay Charts (9-position, N-repeat overlays)
- Flatten Preview (original + regression + flattened + histogram)
- Saturation Trend Charts
- Wafer Map Heatmap
- Best-5 Window comparison
"""
from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for embedding in Qt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from ..core.data_loader import RecipeData, POSITION_LABELS, POSITION_GRID
from ..core.analyzer import AnalysisResult
from ..core.flatten import FlattenResult

# --- Color Scheme ---
COLORS = {
    "bg": "#1e1e2e",
    "fg": "#cdd6f4",
    "grid": "#45475a",
    "accent": "#89b4fa",
    "accent2": "#74c7ec",
    "accent3": "#94e2d5",
    "green": "#a6e3a1",
    "red": "#f38ba8",
    "yellow": "#f9e2af",
    "overlay": [
        "#89b4fa", "#74c7ec", "#94e2d5", "#a6e3a1", "#f9e2af",
        "#fab387", "#f38ba8", "#cba6f7", "#f5c2e7", "#89dceb",
        "#b4befe", "#f5e0dc", "#eba0ac", "#a6d189", "#e78284",
        "#ef9f76", "#81c8be", "#ca9ee6", "#e5c890", "#babbf1",
    ],
}


def _apply_dark_theme(ax, fig=None):
    """Apply dark theme to axes and figure."""
    if fig:
        fig.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["fg"], labelsize=8)
    ax.xaxis.label.set_color(COLORS["fg"])
    ax.yaxis.label.set_color(COLORS["fg"])
    ax.title.set_color(COLORS["fg"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], alpha=0.3, linewidth=0.5)


def create_profile_overlay_figure(recipe: RecipeData,
                                   figsize: tuple = (16, 12)) -> Figure:
    """Create 3×3 grid of profile overlay charts (one per position).

    Each subplot shows all repeat profiles overlaid for that position.
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle(f"Profile Overlay — {recipe.range_label}",
                 color=COLORS["fg"], fontsize=14, fontweight="bold")

    for pos in POSITION_LABELS:
        row, col = POSITION_GRID[pos]
        ax = axes[row][col]
        _apply_dark_theme(ax, fig if (row == 0 and col == 0) else None)

        profiles_found = False
        for i, repeat in enumerate(recipe.repeats):
            if pos in repeat.profiles:
                prof = repeat.profiles[pos]
                color = COLORS["overlay"][i % len(COLORS["overlay"])]
                ax.plot(prof.x_mm, prof.z_nm, color=color, alpha=0.6,
                        linewidth=0.5, label=f"R{repeat.repeat_no}")
                profiles_found = True

        ax.set_title(f"{pos} (Repeats: {recipe.repeat_count})",
                     color=COLORS["fg"], fontsize=9)
        ax.set_xlabel("mm", fontsize=7)
        ax.set_ylabel("nm", fontsize=7)

        if not profiles_found:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                    ha="center", va="center", color=COLORS["fg"], fontsize=10)

    fig.set_facecolor(COLORS["bg"])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def create_flatten_preview_figure(flatten_result: FlattenResult,
                                   x_mm: np.ndarray,
                                   figsize: tuple = (10, 8)) -> Figure:
    """Create XEI-style flatten preview with original, flattened, and histogram.

    Layout:
        Top:    Original profile
        Bottom: Flattened + regression curve + histogram
    """
    fig = plt.figure(figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    fig.suptitle(f"Flatten — Order {flatten_result.order}",
                 color=COLORS["fg"], fontsize=12, fontweight="bold")

    # Top: Original
    ax1 = fig.add_axes([0.08, 0.55, 0.86, 0.38])
    _apply_dark_theme(ax1)
    ax1.plot(x_mm, flatten_result.original, color=COLORS["accent"], linewidth=0.5)
    ax1.set_ylabel("nm", fontsize=9, color=COLORS["fg"])
    ax1.set_title("Original", fontsize=10, color=COLORS["fg"])

    # Bottom left: Flattened + regression
    ax2 = fig.add_axes([0.08, 0.08, 0.58, 0.38])
    _apply_dark_theme(ax2)
    ax2.plot(x_mm, flatten_result.flattened, color=COLORS["accent"],
             linewidth=0.5, alpha=0.7, label="Flattened")
    ax2.plot(x_mm, flatten_result.regression - flatten_result.original.mean(),
             color=COLORS["red"], linewidth=1.2, alpha=0.8, label="Regression")
    ax2.set_xlabel("mm", fontsize=9, color=COLORS["fg"])
    ax2.set_ylabel("nm", fontsize=9, color=COLORS["fg"])
    ax2.set_title("Parameters", fontsize=10, color=COLORS["fg"])
    ax2.legend(fontsize=7, facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["fg"])

    # Bottom right: Histogram
    ax3 = fig.add_axes([0.72, 0.08, 0.22, 0.38])
    _apply_dark_theme(ax3)
    ax3.hist(flatten_result.flattened, bins=50, orientation="horizontal",
             color=COLORS["yellow"], alpha=0.7, edgecolor="none")
    ax3.set_xlabel("Count", fontsize=8, color=COLORS["fg"])

    return fig


def create_saturation_trend_figure(result: AnalysisResult,
                                    figsize: tuple = (10, 6)) -> Figure:
    """Show Rep. 1σ Mean trend as repeat count increases."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    _apply_dark_theme(ax, fig)

    if result.all_windows:
        x = [w.start_index + 1 for w in result.all_windows]
        y = [w.mean_rep_1sigma for w in result.all_windows]
        ax.plot(x, y, 'o-', color=COLORS["accent"], linewidth=2, markersize=8)

        # Highlight best window
        if result.best_window:
            bw = result.best_window
            ax.axvline(x=bw.start_index + 1, color=COLORS["green"],
                       linestyle="--", alpha=0.7, label=f"Best: R{bw.repeat_range}")
            ax.scatter([bw.start_index + 1], [bw.mean_rep_1sigma],
                       color=COLORS["green"], s=150, zorder=5, marker="*")

        # Spec line
        if result.spec_limit:
            ax.axhline(y=result.spec_limit, color=COLORS["red"],
                       linestyle=":", linewidth=2, label=f"Spec: {result.spec_limit} nm")

        ax.set_xlabel("Window Start (Repeat #)", fontsize=10, color=COLORS["fg"])
        ax.set_ylabel("Mean Rep. 1σ (nm)", fontsize=10, color=COLORS["fg"])
        ax.set_title(f"Saturation Trend — {result.range_label}",
                     fontsize=12, color=COLORS["fg"], fontweight="bold")
        ax.legend(fontsize=9, facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
                  labelcolor=COLORS["fg"])

    fig.tight_layout()
    return fig


def create_wafer_map_figure(result: AnalysisResult,
                             metric: str = "opm_max",
                             figsize: tuple = (8, 7)) -> Figure:
    """Create 3×3 wafer map heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    _apply_dark_theme(ax, fig)

    data = np.full((3, 3), np.nan)
    source = result.best_window.positions if result.best_window else result.all_positions

    for pos, pr in source.items():
        if pos in POSITION_GRID:
            r, c = POSITION_GRID[pos]
            val = getattr(pr, metric, pr.opm_max)
            data[r][c] = val

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="equal",
                   interpolation="nearest")

    # Labels
    for pos in POSITION_LABELS:
        if pos in POSITION_GRID:
            r, c = POSITION_GRID[pos]
            val = data[r][c]
            label = f"{pos}\n{val:.2f}" if not np.isnan(val) else pos
            ax.text(c, r, label, ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Left", "Center", "Right"], color=COLORS["fg"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Top", "Middle", "Bottom"], color=COLORS["fg"])

    metric_label = metric.replace("_", " ").title()
    ax.set_title(f"Wafer Map — {result.range_label} ({metric_label})",
                 fontsize=12, color=COLORS["fg"], fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=COLORS["fg"])
    cbar.set_label("nm", color=COLORS["fg"])

    fig.tight_layout()
    return fig


def create_best5_comparison_figure(result: AnalysisResult,
                                    figsize: tuple = (12, 6)) -> Figure:
    """Compare Best-5 Window vs All Repeats statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    _apply_dark_theme(ax1, fig)
    _apply_dark_theme(ax2)

    positions = list(result.all_positions.keys())
    x = np.arange(len(positions))
    width = 0.35

    # Left: Rep. Max comparison
    all_rep_max = [result.all_positions[p].rep_max for p in positions]
    ax1.bar(x - width/2, all_rep_max, width, color=COLORS["accent"], alpha=0.7,
            label="All Repeats")

    if result.best_window:
        bw_rep_max = [result.best_window.positions.get(p, result.all_positions[p]).rep_max
                      for p in positions]
        ax1.bar(x + width/2, bw_rep_max, width, color=COLORS["green"], alpha=0.7,
                label=f"Best-5 (R{result.best_window.repeat_range})")

    ax1.set_xlabel("Position", fontsize=10, color=COLORS["fg"])
    ax1.set_ylabel("Rep. Max (nm)", fontsize=10, color=COLORS["fg"])
    ax1.set_title("Rep. Max Comparison", fontsize=11, color=COLORS["fg"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions, fontsize=7)
    ax1.legend(fontsize=8, facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["fg"])

    # Right: Rep. 1σ comparison
    all_rep_sigma = [result.all_positions[p].rep_1sigma for p in positions]
    ax2.bar(x - width/2, all_rep_sigma, width, color=COLORS["accent"], alpha=0.7,
            label="All Repeats")

    if result.best_window:
        bw_rep_sigma = [result.best_window.positions.get(p, result.all_positions[p]).rep_1sigma
                        for p in positions]
        ax2.bar(x + width/2, bw_rep_sigma, width, color=COLORS["green"], alpha=0.7,
                label=f"Best-5 (R{result.best_window.repeat_range})")

    if result.spec_limit:
        ax2.axhline(y=result.spec_limit, color=COLORS["red"],
                     linestyle=":", linewidth=2, label=f"Spec: {result.spec_limit} nm")

    ax2.set_xlabel("Position", fontsize=10, color=COLORS["fg"])
    ax2.set_ylabel("Rep. 1σ (nm)", fontsize=10, color=COLORS["fg"])
    ax2.set_title("Rep. 1σ Comparison", fontsize=11, color=COLORS["fg"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions, fontsize=7)
    ax2.legend(fontsize=8, facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["fg"])

    fig.suptitle(f"Best-5 Window Analysis — {result.range_label}",
                 fontsize=13, color=COLORS["fg"], fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig
