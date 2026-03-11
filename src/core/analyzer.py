"""OPM Repeatability analyzer.

Calculates OPM, Repeatability statistics, Best-5 Window selection,
and Spec pass/fail judgment.

Key Metrics:
    - OPM:  Max - Min of a single profile (nm)
    - Rep. Max:  Maximum OPM across repeats for a position
    - Rep. 1σ:   Standard deviation of OPM across repeats for a position
    - OPM Max:   Maximum of averaged profile's range across repeats
    - OPM 1σ:    Standard deviation of averaged profile range across repeats
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .data_loader import RecipeData, POSITION_LABELS


# Spec limits (nm) - OPM Repeatability
SPEC_REPEATABILITY = {
    25: 12.9,
    10: 5.6,
    5: 3.3,
    1: 1.6,
}

# Spec limits (nm) - Max OPM
SPEC_MAX_OPM_DW = {   # Double Walled AE
    25: 250.0,
    10: 100.0,
    5: 50.0,
    1: 18.0,
}

SPEC_MAX_OPM_ISO = {   # Isolated AE
    25: 200.0,
    10: 80.0,
    5: 40.0,
    1: 13.0,
}


@dataclass
class PositionResult:
    """Analysis result for a single position across repeats."""
    position: str           # e.g., "1_LT"
    opm_values: np.ndarray  # OPM (nm) per repeat
    rep_max: float          # Max of OPM across repeats
    rep_1sigma: float       # Stdev of OPM across repeats
    opm_max: float          # Max of averaged profile range
    opm_1sigma: float       # Stdev of averaged profile range
    repeat_count: int       # Number of valid repeats


@dataclass
class WindowResult:
    """Result for a single Best-5 window evaluation."""
    start_index: int                    # 0-based start index
    end_index: int                      # 0-based end index (exclusive)
    repeat_range: str                   # e.g., "3-7"
    positions: dict[str, PositionResult]
    mean_rep_1sigma: float              # Mean of Rep. 1σ across positions
    max_rep_max: float                  # Max of Rep. Max across positions
    max_opm_max: float                  # Max of OPM Max across positions


@dataclass
class AnalysisResult:
    """Complete analysis result for a recipe."""
    range_mm: int
    range_label: str
    total_repeats: int

    # Per-position results using ALL repeats
    all_positions: dict[str, PositionResult]

    # Best-5 Window result
    best_window: Optional[WindowResult] = None

    # All window evaluations (for trend display)
    all_windows: list[WindowResult] = field(default_factory=list)

    # Summary statistics
    mean_rep_max: float = 0.0
    stdev_rep_max: float = 0.0
    max_rep_max: float = 0.0
    mean_opm_max: float = 0.0

    # Spec judgment
    spec_limit: Optional[float] = None
    spec_pass: Optional[bool] = None

    @property
    def rms_rep_max(self) -> float:
        """RMS of Rep. Max values across positions."""
        vals = [p.rep_max for p in self.all_positions.values()]
        return float(np.sqrt(np.mean(np.array(vals) ** 2)))


def _compute_position_result(position: str,
                              profiles_z: list[np.ndarray],
                              opm_per_repeat: list[float]) -> PositionResult:
    """Compute statistics for one position across repeats.

    Metrics:
        - Rep. Max: Max of pixel-wise range across repeats (repeatability)
        - Rep. 1σ:  Stdev of pixel-wise range across repeats
        - OPM Max:  Max of per-profile Range (Max-Min) values
        - OPM 1σ:   Stdev of per-profile Range values
    """
    opm_arr = np.array(opm_per_repeat, dtype=np.float64)

    # Pixel-wise repeatability: for each pixel, compute range across repeats
    if len(profiles_z) >= 2:
        stack = np.array(profiles_z, dtype=np.float64)  # (repeats, pixels)
        pixel_range = stack.max(axis=0) - stack.min(axis=0)  # (pixels,)
        rep_max = float(pixel_range.max())
        rep_1sigma = float(pixel_range.std(ddof=0))
    else:
        rep_max = 0.0
        rep_1sigma = 0.0

    return PositionResult(
        position=position,
        opm_values=opm_arr,
        rep_max=rep_max,
        rep_1sigma=rep_1sigma,
        opm_max=float(opm_arr.max()),
        opm_1sigma=float(opm_arr.std(ddof=0)),
        repeat_count=len(opm_per_repeat),
    )


def _evaluate_window(recipe: RecipeData, start: int, count: int) -> Optional[WindowResult]:
    """Evaluate a single sliding window of consecutive repeats."""
    end = start + count
    if end > len(recipe.repeats):
        return None

    window_repeats = recipe.repeats[start:end]
    positions = {}

    for pos in POSITION_LABELS:
        opm_values = []
        profiles_z = []
        for repeat in window_repeats:
            if pos in repeat.profiles:
                profile = repeat.profiles[pos]
                opm_values.append(profile.opm_nm)
                profiles_z.append(profile.z_nm)

        if opm_values:
            positions[pos] = _compute_position_result(pos, profiles_z, opm_values)

    if not positions:
        return None

    mean_rep_1sigma = float(np.mean([p.rep_1sigma for p in positions.values()]))
    max_rep_max = float(max(p.rep_max for p in positions.values()))
    max_opm_max = float(max(p.opm_max for p in positions.values()))

    return WindowResult(
        start_index=start,
        end_index=end,
        repeat_range=f"{start+1}-{end}",
        positions=positions,
        mean_rep_1sigma=mean_rep_1sigma,
        max_rep_max=max_rep_max,
        max_opm_max=max_opm_max,
    )


def analyze_recipe(recipe: RecipeData, window_size: int = 5) -> AnalysisResult:
    """Perform full OPM Repeatability analysis on a recipe.

    Args:
        recipe: RecipeData with loaded profiles.
        window_size: Number of consecutive repeats for Best-5 window.

    Returns:
        AnalysisResult with per-position stats, Best-5 window, and spec judgment.
    """
    n_repeats = recipe.repeat_count

    # --- Compute ALL-repeat statistics per position ---
    all_positions = {}
    for pos in POSITION_LABELS:
        opm_values = []
        profiles_z = []
        for repeat in recipe.repeats:
            if pos in repeat.profiles:
                opm_values.append(repeat.profiles[pos].opm_nm)
                profiles_z.append(repeat.profiles[pos].z_nm)

        if opm_values:
            all_positions[pos] = _compute_position_result(pos, profiles_z, opm_values)

    # --- Evaluate ALL sliding windows ---
    all_windows = []
    if n_repeats >= window_size:
        for start in range(n_repeats - window_size + 1):
            w = _evaluate_window(recipe, start, window_size)
            if w:
                all_windows.append(w)

    # --- Select Best-5 Window (minimum mean Rep. 1σ) ---
    best_window = None
    if all_windows:
        best_window = min(all_windows, key=lambda w: w.mean_rep_1sigma)

    # --- Summary statistics (from all repeats) ---
    rep_maxes = [p.rep_max for p in all_positions.values()]
    opm_maxes = [p.opm_max for p in all_positions.values()]

    mean_rep_max = float(np.mean(rep_maxes)) if rep_maxes else 0.0
    stdev_rep_max = float(np.std(rep_maxes, ddof=0)) if rep_maxes else 0.0
    max_rep_max_val = float(max(rep_maxes)) if rep_maxes else 0.0
    mean_opm_max = float(np.mean(opm_maxes)) if opm_maxes else 0.0

    # --- Spec judgment ---
    range_mm = recipe.range_mm
    spec_limit = SPEC_REPEATABILITY.get(range_mm)
    spec_pass = None
    if spec_limit is not None and best_window is not None:
        # Spec is based on Rep. 1σ Mean from the best window
        spec_pass = best_window.mean_rep_1sigma <= spec_limit

    return AnalysisResult(
        range_mm=range_mm,
        range_label=recipe.range_label,
        total_repeats=n_repeats,
        all_positions=all_positions,
        best_window=best_window,
        all_windows=all_windows,
        mean_rep_max=mean_rep_max,
        stdev_rep_max=stdev_rep_max,
        max_rep_max=max_rep_max_val,
        mean_opm_max=mean_opm_max,
        spec_limit=spec_limit,
        spec_pass=spec_pass,
    )


def get_summary_table(result: AnalysisResult,
                      use_best_window: bool = True) -> list[dict]:
    """Generate summary table rows (AFP-compatible format).

    Returns list of dicts with: Range, Position, Rep. Max, Rep. 1σ, OPM Max, OPM 1σ.
    """
    source = result.best_window.positions if (use_best_window and result.best_window) else result.all_positions

    rows = []
    for pos in POSITION_LABELS:
        if pos in source:
            p = source[pos]
            rows.append({
                "Range": result.range_label,
                "Position": pos,
                "Rep. Max (nm)": round(p.rep_max, 3),
                "Rep. 1σ (nm)": round(p.rep_1sigma, 3),
                "OPM Max (nm)": round(p.opm_max, 3),
                "OPM 1σ (nm)": round(p.opm_1sigma, 3),
            })

    # Total row
    if rows:
        rep_maxes = [r["Rep. Max (nm)"] for r in rows]
        rep_sigmas = [r["Rep. 1σ (nm)"] for r in rows]
        opm_maxes = [r["OPM Max (nm)"] for r in rows]
        opm_sigmas = [r["OPM 1σ (nm)"] for r in rows]

        rows.append({"Range": "Total", "Position": "Mean",
                      "Rep. Max (nm)": round(float(np.mean(rep_maxes)), 3),
                      "Rep. 1σ (nm)": round(float(np.mean(rep_sigmas)), 3),
                      "OPM Max (nm)": round(float(np.mean(opm_maxes)), 3),
                      "OPM 1σ (nm)": round(float(np.mean(opm_sigmas)), 3)})
        rows.append({"Range": "Total", "Position": "Stdev",
                      "Rep. Max (nm)": round(float(np.std(rep_maxes, ddof=0)), 3),
                      "Rep. 1σ (nm)": round(float(np.std(rep_sigmas, ddof=0)), 3),
                      "OPM Max (nm)": round(float(np.std(opm_maxes, ddof=0)), 3),
                      "OPM 1σ (nm)": round(float(np.std(opm_sigmas, ddof=0)), 3)})
        rows.append({"Range": "Total", "Position": "Max",
                      "Rep. Max (nm)": round(float(max(rep_maxes)), 3),
                      "Rep. 1σ (nm)": "-",
                      "OPM Max (nm)": round(float(max(opm_maxes)), 3),
                      "OPM 1σ (nm)": "-"})
        rms_rep = float(np.sqrt(np.mean(np.array(rep_maxes) ** 2)))
        rms_opm = float(np.sqrt(np.mean(np.array(opm_maxes) ** 2)))
        rows.append({"Range": "Total", "Position": "RMS",
                      "Rep. Max (nm)": round(rms_rep, 3),
                      "Rep. 1σ (nm)": round(float(np.mean(rep_sigmas)), 3),
                      "OPM Max (nm)": round(rms_opm, 3),
                      "OPM 1σ (nm)": round(float(np.mean(opm_sigmas)), 3)})

    return rows
