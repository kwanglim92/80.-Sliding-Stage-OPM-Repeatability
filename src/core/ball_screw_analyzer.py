"""Ball Screw Pitch analyzer for Sliding Stage OPM data.

Analyzes Dishing and Erosion from TIFF profile data using
12th-order polynomial flattening with XE-compatible CMP Percentile method.

Analysis Method:
    1. 12th-order Polynomial Flatten (full data, no edge exclusion)
    2. CMP Percentile extraction:
       - Base zone  : Offset 10%, Range 10% → ErosionTop 50th percentile
       - Interest zone: Offset 50%, Range 100% → DishingTop 100th / DishingBottom 1st percentile
    3. Erosion (nm) = ErosionTop - DishingBottom
    4. Dishing (nm) = DishingTop - DishingBottom

Spec (Dishing):
    AL  : ≤ 6.0 nm
    SUS : ≤ 4.5 nm

Spec Judgment: MAX Dishing across all Repeats per Position (conservative / B-plan).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .tiff_reader import ProfileData, read_profile
from .data_loader import RecipeData, POSITION_LABELS


# ─── Spec limits (nm) — Dishing ───────────────────────────────────────────────

SPEC_DISHING: dict[str, float] = {
    "AL": 6.0,
    "SUS": 4.5,
}

# ─── Analysis Parameters ───────────────────────────────────────────────────────

FLATTEN_ORDER = 12
# Base zone: front ~10% of profile for Erosion reference
BASE_OFFSET_PCT = 10.0    # start at 10% of scan length
BASE_RANGE_PCT = 10.0     # use 10% width
EROSION_TOP_PCT = 50.0    # ErosionTop = 50th percentile within base zone

# Interest zone: full profile (0% offset, 100% range)
INTEREST_OFFSET_PCT = 0.0
INTEREST_RANGE_PCT = 100.0
DISHING_TOP_PCT = 99.0    # DishingTop  = 99th percentile  (matches XE CMP Percentile)
DISHING_BOTTOM_PCT = 1.0  # DishingBottom = 1st percentile

# Text CSV subdirectory name under each recipe folder (XE-exported)
_TEXT_SUBDIR = "Text"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class BallScrewPointResult:
    """Analysis result for a single measurement point."""
    point_no: int         # Raw point number (1-based, 1 = stabilization)
    position: str         # "1_LT_stab" for pt1, "1_LT"~"9_RB" for pt2~10
    is_stabilization: bool
    erosion_nm: float
    dishing_nm: float
    spec_limit: Optional[float]    # None for stabilization point
    spec_pass: Optional[bool]      # None for stabilization point


@dataclass
class BallScrewRepeatResult:
    """Analysis results for one Repeat (all 10 raw points)."""
    repeat_no: int
    repeat_dir: str                        # Directory name for identification
    points: list[BallScrewPointResult]     # Length 10 (includes stabilization)

    @property
    def valid_points(self) -> list[BallScrewPointResult]:
        """Points excluding stabilization (for Spec judgment)."""
        return [p for p in self.points if not p.is_stabilization]


@dataclass
class PositionStats:
    """Cross-repeat statistics for one Position."""
    position: str
    repeat_values: list[float]     # Dishing per repeat (5 values)
    mean_nm: float
    stdev_nm: float
    max_nm: float
    min_nm: float
    range_nm: float                # Max - Min
    spec_limit: float
    spec_pass: bool                # Based on MAX (B-plan)


@dataclass
class BallScrewAnalysisResult:
    """Complete Ball Screw Pitch analysis result."""
    range_mm: int
    material: str                               # "AL" or "SUS"
    spec_limit: float                           # 6.0 or 4.5 nm
    signal_source: str                          # "Height" or "Z Drive"
    all_repeats: list[BallScrewRepeatResult]   # All 5 Repeats
    position_stats: dict[str, PositionStats]   # Per-position cross-repeat stats
    overall_pass: bool                          # True if ALL positions PASS


# ─── Core Calculation ──────────────────────────────────────────────────────────

def _polynomial_flatten(z_data: np.ndarray, order: int = 12) -> np.ndarray:
    """Apply nth-order polynomial flatten to profile data.

    Uses normalized X for numerical stability (same as FlattenProcessor).
    No edge exclusion (full data used for Ball Screw Pitch analysis).
    """
    n = len(z_data)
    x_data = np.arange(n, dtype=np.float64)

    if order == 0:
        return z_data - z_data.mean()

    x_mean = x_data.mean()
    x_std = x_data.std()
    if x_std == 0:
        x_std = 1.0
    x_norm = (x_data - x_mean) / x_std

    coefficients = np.polyfit(x_norm, z_data, order)
    regression = np.polyval(coefficients, x_norm)
    return z_data - regression


def compute_dishing(
    z_data: np.ndarray,
    flatten_order: int = FLATTEN_ORDER,
    base_offset_pct: float = BASE_OFFSET_PCT,
    base_range_pct: float = BASE_RANGE_PCT,
    erosion_top_pct: float = EROSION_TOP_PCT,
    interest_offset_pct: float = INTEREST_OFFSET_PCT,
    interest_range_pct: float = INTEREST_RANGE_PCT,
    dishing_top_pct: float = DISHING_TOP_PCT,
    dishing_bottom_pct: float = DISHING_BOTTOM_PCT,
    pre_flattened: bool = False,
) -> tuple[float, float]:
    """Compute Erosion and Dishing from a Z profile using CMP Percentile method.

    Replicates XE Software CMP Percentile analysis:
        1. Flatten with polynomial regression (skip if pre_flattened=True)
        2. Extract Base zone → ErosionTop percentile
        3. Extract Interest zone → DishingTop / DishingBottom percentiles
        4. Erosion = DishingBottom - ErosionTop  (negative = surface eroded below base)
        5. Dishing = DishingTop - DishingBottom

    Args:
        z_data: Z-height profile in nm (1D array).
        flatten_order: Polynomial order for flattening (default 12).
        base_offset_pct: Base zone start offset (% of total length).
        base_range_pct: Base zone width (% of total length).
        erosion_top_pct: Percentile within base zone for Erosion reference.
        interest_offset_pct: Interest zone start offset (%).
        interest_range_pct: Interest zone width (%).
        dishing_top_pct: Top percentile within interest zone for DishingTop.
        dishing_bottom_pct: Bottom percentile within interest zone for DishingBottom.
        pre_flattened: If True, skip flatten step (data already processed by XE SW).

    Returns:
        (erosion_nm, dishing_nm) tuple.
    """
    n = len(z_data)
    if n < flatten_order + 2 and not pre_flattened:
        raise ValueError(f"Not enough data points ({n}) for order-{flatten_order} fit.")

    # --- Step 1: Flatten (skip if data is already XE-flattened) ---
    if pre_flattened:
        flattened = z_data
    else:
        flattened = _polynomial_flatten(z_data, order=flatten_order)

    # --- Step 2: Base Zone → ErosionTop ---
    base_start = int(n * base_offset_pct / 100.0)
    base_end = min(n, base_start + int(n * base_range_pct / 100.0))
    base_zone = flattened[base_start:base_end]
    erosion_top_val = float(np.percentile(base_zone, erosion_top_pct))

    # --- Step 3: Interest Zone → DishingTop / DishingBottom ---
    int_start = int(n * interest_offset_pct / 100.0)
    int_end = min(n, int_start + int(n * interest_range_pct / 100.0))
    if int_end <= int_start:
        int_end = n
    interest_zone = flattened[int_start:int_end]
    dishing_top_val = float(np.percentile(interest_zone, dishing_top_pct))
    dishing_bottom_val = float(np.percentile(interest_zone, dishing_bottom_pct))

    # --- Step 4 & 5: Erosion and Dishing ---
    # Erosion sign convention matches XE SW: negative = surface is lower than base
    # Erosion = DishingBottom - ErosionTop  (negative when surface eroded below base)
    erosion_nm = dishing_bottom_val - erosion_top_val
    dishing_nm = dishing_top_val - dishing_bottom_val

    return erosion_nm, dishing_nm


# ─── Main Analysis Function ─────────────────────────────────────────────────────────────

def analyze_ball_screw(
    recipe: RecipeData,
    signal_source: str = "Height",
    material: str = "AL",
) -> BallScrewAnalysisResult:
    """Perform Ball Screw Pitch analysis across all Repeats.

    Analyzes all 10 raw points per Repeat (including stabilization Point 1).
    Position mapping:
        Point 1 → "1_LT_stab" (stabilization, excluded from Spec judgment)
        Point 2 → "1_LT"
        Point 3 → "2_CT"
        ...
        Point 10 → "9_RB"

    Spec judgment uses MAX Dishing across Repeats per Position (B-plan).

    Args:
        recipe: RecipeData with loaded profiles.
        signal_source: "Height" or "Z Drive".
        material: "AL" (≤6.0 nm) or "SUS" (≤4.5 nm).

    Returns:
        BallScrewAnalysisResult with per-repeat and cross-repeat statistics.
    """
    spec_limit = SPEC_DISHING.get(material, SPEC_DISHING["AL"])
    all_repeats: list[BallScrewRepeatResult] = []

    for repeat in recipe.repeats:
        point_results: list[BallScrewPointResult] = []

        # Build a list of all raw points in order: point_no 1..10
        # Point 1 = stabilization (position 1_LT_stab)
        # Points 2..10 = POSITION_LABELS[0..8] = 1_LT..9_RB

        # Collect available profiles (sorted by filename to get original order)
        # Use the profiles dict already loaded by the data_loader
        raw_tiff_dir = repeat.directory
        text_dir = raw_tiff_dir / _TEXT_SUBDIR  # XE-exported Text CSV folder

        for raw_pt_no in range(1, 11):  # 1 to 10
            is_stab = (raw_pt_no == 1)

            if is_stab:
                position = "1_LT_stab"
                pos_key = None  # Not in profiles dict (stabilization excluded by data_loader)
            else:
                pos_idx = raw_pt_no - 2  # 0-based index into POSITION_LABELS
                position = POSITION_LABELS[pos_idx]
                pos_key = position

            # Priority: XE Text CSV (already XE-flattened) > TIFF profiles > TIFF direct load
            z_data: Optional[np.ndarray] = None
            use_text_csv = False

            # 1) Try XE Text CSV (highest accuracy vs XE SW output)
            if text_dir.is_dir():
                csv_pattern = f"*_{raw_pt_no:04d}_{signal_source}.csv"
                csv_files = list(text_dir.glob(csv_pattern))
                if csv_files:
                    try:
                        import csv as _csv
                        ys = []
                        with open(csv_files[0], 'r', encoding='utf-8-sig') as f:
                            reader = _csv.reader(f)
                            next(reader)  # skip header
                            for row in reader:
                                if len(row) >= 2:
                                    try:
                                        ys.append(float(row[1]))
                                    except ValueError:
                                        pass
                        if ys:
                            z_data = np.array(ys, dtype=np.float64)
                            use_text_csv = True
                    except Exception:
                        z_data = None

            # 2) Fall back to already-loaded TIFF profiles
            if z_data is None and pos_key is not None and pos_key in repeat.profiles:
                z_data = repeat.profiles[pos_key].z_nm

            # 3) Last resort: load TIFF directly (includes stabilization point)
            if z_data is None:
                suffix = f"_{signal_source}.tiff"
                pattern = f"*_{raw_pt_no:04d}_{signal_source}.tiff"
                tiff_files = list(raw_tiff_dir.glob(pattern))
                if tiff_files:
                    try:
                        profile = read_profile(tiff_files[0])
                        z_data = profile.z_nm
                    except Exception:
                        z_data = None

            if z_data is not None and len(z_data) > FLATTEN_ORDER + 2:
                try:
                    # XE Text CSV is already flattened by XE SW → use pre_flattened=True
                    erosion_nm, dishing_nm = compute_dishing(
                        z_data, pre_flattened=use_text_csv)
                    if is_stab:
                        spec_pass_val = None
                        spec_lim_val = None
                    else:
                        spec_pass_val = dishing_nm <= spec_limit
                        spec_lim_val = spec_limit
                except Exception:
                    erosion_nm, dishing_nm = float("nan"), float("nan")
                    spec_pass_val = None
                    spec_lim_val = None
            else:
                erosion_nm, dishing_nm = float("nan"), float("nan")
                spec_pass_val = None
                spec_lim_val = None

            point_results.append(BallScrewPointResult(
                point_no=raw_pt_no,
                position=position,
                is_stabilization=is_stab,
                erosion_nm=erosion_nm,
                dishing_nm=dishing_nm,
                spec_limit=spec_lim_val,
                spec_pass=spec_pass_val,
            ))

        all_repeats.append(BallScrewRepeatResult(
            repeat_no=repeat.repeat_no,
            repeat_dir=str(repeat.directory.name),
            points=point_results,
        ))

    # ─── Cross-Repeat Statistics per Position ─────────────────────────────────
    position_stats: dict[str, PositionStats] = {}

    for pos in POSITION_LABELS:
        repeat_dishing_values = []
        for rep_result in all_repeats:
            matching = [p for p in rep_result.points
                        if p.position == pos and not np.isnan(p.dishing_nm)]
            if matching:
                repeat_dishing_values.append(matching[0].dishing_nm)
            else:
                repeat_dishing_values.append(float("nan"))

        valid_vals = [v for v in repeat_dishing_values if not np.isnan(v)]
        if valid_vals:
            mean_nm = float(np.mean(valid_vals))
            stdev_nm = float(np.std(valid_vals, ddof=0))
            max_nm = float(np.max(valid_vals))
            min_nm = float(np.min(valid_vals))
            range_nm = max_nm - min_nm
            # B-plan: Spec judgment based on MAX
            spec_pass = max_nm <= spec_limit
        else:
            mean_nm = stdev_nm = max_nm = min_nm = range_nm = float("nan")
            spec_pass = False

        position_stats[pos] = PositionStats(
            position=pos,
            repeat_values=repeat_dishing_values,
            mean_nm=mean_nm,
            stdev_nm=stdev_nm,
            max_nm=max_nm,
            min_nm=min_nm,
            range_nm=range_nm,
            spec_limit=spec_limit,
            spec_pass=spec_pass,
        )

    # ─── Overall Pass/Fail ────────────────────────────────────────────────────
    overall_pass = all(
        s.spec_pass for s in position_stats.values()
        if not np.isnan(s.max_nm)
    )

    return BallScrewAnalysisResult(
        range_mm=recipe.range_mm,
        material=material,
        spec_limit=spec_limit,
        signal_source=signal_source,
        all_repeats=all_repeats,
        position_stats=position_stats,
        overall_pass=overall_pass,
    )


def get_dishing_matrix(result: BallScrewAnalysisResult,
                       include_stabilization: bool = False
                       ) -> tuple[list[str], list[str], np.ndarray]:
    """Build a Position × Repeat Dishing matrix for display.

    Returns:
        positions: List of position labels (rows).
        repeat_labels: List of repeat labels (columns).
        matrix: 2D array (n_positions × n_repeats) of Dishing values.
    """
    positions = []
    for rep in result.all_repeats:
        for pt in rep.points:
            if include_stabilization or not pt.is_stabilization:
                if pt.position not in positions:
                    positions.append(pt.position)

    repeat_labels = [f"Rep.{r.repeat_no}" for r in result.all_repeats]
    n_pos = len(positions)
    n_rep = len(result.all_repeats)
    matrix = np.full((n_pos, n_rep), fill_value=np.nan)

    pos_idx = {p: i for i, p in enumerate(positions)}
    for rep_i, rep in enumerate(result.all_repeats):
        for pt in rep.points:
            if pt.position in pos_idx:
                matrix[pos_idx[pt.position], rep_i] = pt.dishing_nm

    return positions, repeat_labels, matrix
