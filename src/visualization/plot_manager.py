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
from ..core.analyzer import AnalysisResult, POSITION_GROUPS, resample_profile
from ..core.flatten import FlattenResult, polynomial_flatten

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
                                   figsize: tuple = (16, 12),
                                   scan_info: dict | None = None,
                                   y_scale_mode: str = "auto",
                                   sim_factor: int = 1) -> Figure:
    """Create 3×3 grid of profile overlay charts (one per position).

    Each subplot shows all repeat profiles overlaid for that position.
    Profiles are Order-1 leveled (line subtracted) so endpoints converge to 0.
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)

    # Build suptitle with scan info if available
    title = f"Profile Overlay — {recipe.range_label}"
    if scan_info:
        info_parts = [
            f"{scan_info.get('range_label', '')}",
            f"{scan_info.get('pixels', '')}px",
            f"{scan_info.get('resolution_nm', 0):.0f}nm/px",
            f"{scan_info.get('speed', 0):.2g}mm/s",
            f"SP={scan_info.get('set_point', 0):.1f}",
        ]
        title = f"Profile Overlay — {' | '.join(info_parts)}"

    if sim_factor > 1:
        orig_res = scan_info.get("resolution_nm", 0) if scan_info else 0
        sim_res = orig_res * sim_factor
        title += f"  [Simulated: {sim_res:.0f} nm/px, ×{sim_factor}]"

    fig.suptitle(title, color=COLORS["fg"], fontsize=14, fontweight="bold")

    for pos in POSITION_LABELS:
        row, col = POSITION_GRID[pos]
        ax = axes[row][col]
        _apply_dark_theme(ax, fig if (row == 0 and col == 0) else None)

        profiles_found = False
        for i, repeat in enumerate(recipe.repeats):
            if pos in repeat.profiles:
                prof = repeat.profiles[pos]
                # Apply resampling if simulating lower resolution
                if sim_factor > 1:
                    z_rs = resample_profile(prof.z_nm, sim_factor)
                    x_rs = resample_profile(prof.x_mm, sim_factor)
                    z_leveled = polynomial_flatten(z_rs, x_data=x_rs, order=1)
                    x_plot = x_rs
                else:
                    z_leveled = polynomial_flatten(prof.z_nm, x_data=prof.x_mm, order=1)
                    x_plot = prof.x_mm
                color = COLORS["overlay"][i % len(COLORS["overlay"])]
                ax.plot(x_plot, z_leveled, color=color, alpha=0.6,
                        linewidth=0.5, label=f"R{repeat.repeat_no}")
                profiles_found = True

        ax.set_title(f"{pos} (Repeats: {recipe.repeat_count})",
                     color=COLORS["fg"], fontsize=9)
        ax.set_xlabel("mm", fontsize=7)
        ax.set_ylabel("nm", fontsize=7)

        if not profiles_found:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                    ha="center", va="center", color=COLORS["fg"], fontsize=10)

    if y_scale_mode != "auto":
        _sync_y_axes(axes, y_scale_mode)

    fig.set_facecolor(COLORS["bg"])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _sync_y_axes(axes, mode: str):
    """Synchronize Y-axis limits: 'unified' (all same) or 'group' (per group)."""
    _PADDING = 1.1

    if mode == "unified":
        max_abs = max(
            (np.max(np.abs(line.get_ydata()))
             for row in axes for ax in row for line in ax.get_lines()
             if len(line.get_ydata()) > 0),
            default=0.0)
        if max_abs > 0:
            lim = max_abs * _PADDING
            for row in axes:
                for ax in row:
                    ax.set_ylim(-lim, lim)

    elif mode == "group":
        for group_positions in POSITION_GROUPS.values():
            group_axes = [axes[POSITION_GRID[p][0]][POSITION_GRID[p][1]]
                          for p in group_positions if p in POSITION_GRID]
            max_abs = max(
                (np.max(np.abs(line.get_ydata()))
                 for ax in group_axes for line in ax.get_lines()
                 if len(line.get_ydata()) > 0),
                default=0.0)
            if max_abs > 0:
                lim = max_abs * _PADDING
                for ax in group_axes:
                    ax.set_ylim(-lim, lim)


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

    fig.tight_layout(pad=2.0)
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


def create_resolution_comparison_figure(
        norm_data: dict[str, dict[str, dict]],
        figsize: tuple = (14, 7),
        spec_limits: dict[int, float] | None = None) -> Figure:
    """Create Original vs Normalized OPM comparison across ranges.

    Args:
        norm_data: dict[range_label, dict[position, {original_opm, normalized_opm, ...}]]
        figsize: Figure size.
        spec_limits: Optional {range_mm: limit_nm} for Spec lines on normalized chart.

    Returns:
        matplotlib Figure.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    _apply_dark_theme(ax1, fig)
    _apply_dark_theme(ax2)
    for ax in (ax1, ax2):
        ax.grid(axis='y', color=COLORS["grid"], alpha=0.3)

    range_labels = list(norm_data.keys())
    range_colors = [COLORS["accent"], COLORS["accent2"], COLORS["accent3"], COLORS["yellow"]]

    # --- Left: Position-wise bars grouped by range ---
    positions = ["1_LT", "2_CT", "3_RT", "4_LM", "5_CM", "6_RM", "7_LB", "8_CB", "9_RB"]
    n_ranges = len(range_labels)
    x = np.arange(len(positions))
    bar_width = 0.8 / max(n_ranges, 1)

    for i, rlabel in enumerate(range_labels):
        orig_vals = [norm_data[rlabel].get(p, {}).get("original_opm", 0) for p in positions]
        norm_vals = [norm_data[rlabel].get(p, {}).get("normalized_opm", 0) for p in positions]
        offset = (i - n_ranges / 2 + 0.5) * bar_width
        ax1.bar(x + offset, orig_vals, bar_width * 0.9,
                color=range_colors[i % len(range_colors)], alpha=0.7, label=rlabel)

    ax1.set_xlabel("Position", fontsize=10, color=COLORS["fg"])
    ax1.set_ylabel("OPM Max (nm)", fontsize=10, color=COLORS["fg"])
    ax1.set_title("Original OPM Max per Position", fontsize=11, color=COLORS["fg"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions, fontsize=7)
    ax1.legend(fontsize=8, facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["fg"])

    # --- Right: Normalized bars (same resolution) ---
    for i, rlabel in enumerate(range_labels):
        norm_vals = [norm_data[rlabel].get(p, {}).get("normalized_opm", 0) for p in positions]
        offset = (i - n_ranges / 2 + 0.5) * bar_width
        ax2.bar(x + offset, norm_vals, bar_width * 0.9,
                color=range_colors[i % len(range_colors)], alpha=0.7, label=rlabel)

    ax2.set_xlabel("Position", fontsize=10, color=COLORS["fg"])
    ax2.set_ylabel("OPM Max (nm) — Normalized", fontsize=10, color=COLORS["fg"])
    target_res = max(
        norm_data[rl].get("5_CM", {}).get("original_res", 0)
        for rl in range_labels
    ) if range_labels else 0
    ax2.set_title(f"Normalized OPM Max (target: {target_res:.0f} nm/px)",
                  fontsize=11, color=COLORS["fg"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions, fontsize=7)
    ax2.legend(fontsize=8, facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["fg"])

    # --- Spec lines on normalized chart ---
    if spec_limits:
        for i, rlabel in enumerate(range_labels):
            range_mm = int(rlabel.replace("mm", ""))
            if range_mm in spec_limits:
                spec_val = spec_limits[range_mm]
                ax2.axhline(y=spec_val, color=range_colors[i % len(range_colors)],
                            linestyle="--", alpha=0.5, linewidth=1.0)
                ax2.text(len(positions) - 0.5, spec_val,
                         f" Spec {rlabel}: {spec_val:.0f}nm",
                         color=range_colors[i % len(range_colors)],
                         fontsize=7, va="bottom", ha="right")

    # --- Reduction % annotations on normalized chart ---
    for i, rlabel in enumerate(range_labels):
        orig_vals = [norm_data[rlabel].get(p, {}).get("original_opm", 0) for p in positions]
        norm_vals = [norm_data[rlabel].get(p, {}).get("normalized_opm", 0) for p in positions]
        orig_mean = np.mean([v for v in orig_vals if v > 0]) if any(v > 0 for v in orig_vals) else 0
        norm_mean = np.mean([v for v in norm_vals if v > 0]) if any(v > 0 for v in norm_vals) else 0
        if orig_mean > 0:
            reduction = (orig_mean - norm_mean) / orig_mean * 100
            offset = (i - n_ranges / 2 + 0.5) * bar_width
            max_norm = max(norm_vals) if norm_vals else 0
            ax2.text(x[-1] + offset, max_norm * 1.02, f"{reduction:+.0f}%",
                     color=range_colors[i % len(range_colors)],
                     fontsize=7, fontweight="bold", ha="center", va="bottom")

    fig.suptitle("Cross-Range Resolution Comparison",
                 fontsize=14, color=COLORS["fg"], fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig
