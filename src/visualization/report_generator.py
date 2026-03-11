"""Report generator for exporting analysis results.

Supports:
- Summary Table (Excel/CSV) - AFP format compatible
- Avg. Line CSV / All Line CSV
- Chart images (.png) batch export
- Spec judgment Checklist
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..core.analyzer import AnalysisResult, get_summary_table
from ..core.data_loader import RecipeData, POSITION_LABELS


def export_summary_csv(result: AnalysisResult, output_path: str | Path,
                       use_best_window: bool = True) -> None:
    """Export summary table as CSV."""
    rows = get_summary_table(result, use_best_window=use_best_window)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        if rows:
            headers = list(rows[0].keys())
            f.write(",".join(headers) + "\n")
            for row in rows:
                f.write(",".join(str(row[h]) for h in headers) + "\n")


def export_avg_line_csv(recipe: RecipeData, output_path: str | Path) -> None:
    """Export averaged line profiles per position as CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        # Header
        f.write("X (mm)")
        for pos in POSITION_LABELS:
            f.write(f",{pos} Avg (nm)")
        f.write("\n")

        # Compute averaged profiles
        avg_profiles = {}
        x_mm = None
        for pos in POSITION_LABELS:
            profiles = []
            for repeat in recipe.repeats:
                if pos in repeat.profiles:
                    profiles.append(repeat.profiles[pos].z_nm)
                    if x_mm is None:
                        x_mm = repeat.profiles[pos].x_mm
            if profiles:
                avg_profiles[pos] = np.mean(profiles, axis=0)

        if x_mm is not None:
            for i in range(len(x_mm)):
                f.write(f"{x_mm[i]:.6f}")
                for pos in POSITION_LABELS:
                    if pos in avg_profiles:
                        f.write(f",{avg_profiles[pos][i]:.6f}")
                    else:
                        f.write(",")
                f.write("\n")


def export_all_lines_csv(recipe: RecipeData, output_path: str | Path) -> None:
    """Export all individual line profiles as CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        # Header
        f.write("X (mm)")
        for repeat in recipe.repeats:
            for pos in POSITION_LABELS:
                f.write(f",R{repeat.repeat_no}_{pos} (nm)")
        f.write("\n")

        # Get x_mm from first available profile
        x_mm = None
        for repeat in recipe.repeats:
            for pos, prof in repeat.profiles.items():
                x_mm = prof.x_mm
                break
            if x_mm is not None:
                break

        if x_mm is not None:
            for i in range(len(x_mm)):
                f.write(f"{x_mm[i]:.6f}")
                for repeat in recipe.repeats:
                    for pos in POSITION_LABELS:
                        if pos in repeat.profiles:
                            f.write(f",{repeat.profiles[pos].z_nm[i]:.6f}")
                        else:
                            f.write(",")
                f.write("\n")


def export_checklist(result: AnalysisResult, output_path: str | Path) -> None:
    """Export spec judgment checklist."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(f"Sliding Stage OPM Repeatability Checklist\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Range: {result.range_label}\n")
        f.write(f"Total Repeats: {result.total_repeats}\n\n")

        if result.best_window:
            bw = result.best_window
            f.write(f"Best-5 Window: Repeats {bw.repeat_range}\n")
            f.write(f"Mean Rep. 1σ: {bw.mean_rep_1sigma:.3f} nm\n")
            f.write(f"Max Rep. Max: {bw.max_rep_max:.3f} nm\n\n")

        if result.spec_limit is not None:
            verdict = "PASS ✓" if result.spec_pass else "FAIL ✗"
            f.write(f"Spec Limit: {result.spec_limit} nm\n")
            f.write(f"Judgment: {verdict}\n\n")

        f.write(f"{'Position':<10} {'Rep.Max':>10} {'Rep.1σ':>10} {'OPM Max':>10} {'OPM 1σ':>10}\n")
        f.write("-" * 50 + "\n")

        source = result.best_window.positions if result.best_window else result.all_positions
        for pos in POSITION_LABELS:
            if pos in source:
                p = source[pos]
                f.write(f"{pos:<10} {p.rep_max:>10.3f} {p.rep_1sigma:>10.3f} "
                        f"{p.opm_max:>10.3f} {p.opm_1sigma:>10.3f}\n")
