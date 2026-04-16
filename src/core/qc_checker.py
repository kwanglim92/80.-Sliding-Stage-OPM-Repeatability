"""Data Collection QC Checker for Sliding Stage OPM Repeatability.

Performs 6 integrity checks on loaded recipe data:
    QC-1: Recipe TIFF vs Raw TIFF file matching
    QC-2: Recipe TIFF vs Raw TIFF data equivalence
    QC-3: Scan parameter consistency across repeats/positions
    QC-4: Position completeness (9 pos x N repeats)
    QC-5: Statistical outlier detection (Median +/- 3*MAD)
    QC-6: Pixel count consistency (expected: 8192)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None

from .data_loader import RecipeData, POSITION_LABELS

# Park Systems TIFF tag for raw profile data
_TAG_RAW_DATA = 50434

# Expected pixel count for 1D profile
_EXPECTED_PIXELS = 8192

# MAD normalization constant (consistent estimator of sigma for Gaussian)
_MAD_SCALE = 1.4826


@dataclass
class QCItemResult:
    """Result of a single QC check."""
    check_id: str          # "QC-1" through "QC-6"
    name: str              # Human-readable check name
    status: str            # "PASS", "WARN", "FAIL"
    summary: str           # One-line summary
    details: list[dict] = field(default_factory=list)


@dataclass
class QCResult:
    """Aggregate result of all 6 QC checks."""
    checks: list[QCItemResult]
    overall_status: str    # "PASS", "WARN", "FAIL"
    recipe_label: str
    timestamp: str


def run_qc_checks(recipe: RecipeData, signal_source: str = "Height") -> QCResult:
    """Run all 6 QC checks on a loaded recipe.

    Args:
        recipe: RecipeData with loaded profiles.
        signal_source: "Height" or "Z Drive".

    Returns:
        QCResult with per-check results and overall status.
    """
    checks = [
        _check_file_matching(recipe, signal_source),
        _check_data_equivalence(recipe, signal_source),
        _check_scan_parameters(recipe),
        _check_position_completeness(recipe),
        _check_statistical_outliers(recipe),
        _check_pixel_count(recipe),
    ]

    # Overall: FAIL if any FAIL, WARN if any WARN, else PASS
    statuses = [c.status for c in checks]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    return QCResult(
        checks=checks,
        overall_status=overall,
        recipe_label=recipe.range_label,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ---------------------------------------------------------------------------
# QC-1: File Matching (Recipe TIFF <-> Raw TIFF)
# ---------------------------------------------------------------------------

def _check_file_matching(recipe: RecipeData, signal_source: str) -> QCItemResult:
    """Check that every Recipe TIFF has a matching file in Debug/Raw."""
    details = []
    fail_count = 0
    total_count = 0

    suffix = f"_{signal_source}.tiff"

    for repeat in recipe.repeats:
        raw_dir = repeat.directory / "Debug" / "Raw"

        # Get recipe-level TIFF files
        recipe_files = {f.name for f in repeat.directory.glob(f"*{suffix}")}
        total_count += len(recipe_files)

        if not raw_dir.is_dir():
            # Debug/Raw directory missing entirely
            for fname in sorted(recipe_files):
                details.append({
                    "repeat": repeat.repeat_no,
                    "folder": repeat.directory.name,
                    "file": fname,
                    "status": "FAIL",
                    "detail": "Debug/Raw directory not found",
                })
            fail_count += len(recipe_files)
            continue

        raw_files = {f.name for f in raw_dir.glob(f"*{suffix}")}

        for fname in sorted(recipe_files):
            if fname in raw_files:
                details.append({
                    "repeat": repeat.repeat_no,
                    "folder": repeat.directory.name,
                    "file": fname,
                    "status": "PASS",
                    "detail": "Matched",
                })
            else:
                details.append({
                    "repeat": repeat.repeat_no,
                    "folder": repeat.directory.name,
                    "file": fname,
                    "status": "FAIL",
                    "detail": "Missing in Debug/Raw",
                })
                fail_count += 1

    if fail_count > 0:
        status = "FAIL"
        summary = f"{fail_count}/{total_count} files missing in Debug/Raw"
    else:
        status = "PASS"
        summary = f"All {total_count} files matched"

    return QCItemResult(
        check_id="QC-1",
        name="File Matching",
        status=status,
        summary=summary,
        details=details,
    )


# ---------------------------------------------------------------------------
# QC-2: Data Equivalence (Recipe TIFF vs Raw TIFF after Flatten)
# ---------------------------------------------------------------------------
# Recipe TIFF (root) has internal processing applied, so raw bytes differ
# from Debug/Raw TIFF.  However, after polynomial flatten, results should
# be identical. This check reads Raw TIFFs, applies the same flatten as
# the analyzer, and compares the OPM (Range) values.

def _check_data_equivalence(recipe: RecipeData, signal_source: str) -> QCItemResult:
    """Compare analysis results between Recipe and Raw TIFFs after flatten.

    Reads Raw TIFFs from Debug/Raw, applies Order-2 flatten, and compares
    the resulting Range (Max-Min) with the already-loaded Recipe TIFF profiles.
    """
    from .tiff_reader import read_profile
    from .flatten import polynomial_flatten

    details = []
    fail_count = 0
    total_count = 0
    skip_count = 0

    # Tolerance for Range comparison (nm) — sub-picometer level
    _RANGE_TOL = 0.001

    for repeat in recipe.repeats:
        raw_dir = repeat.directory / "Debug" / "Raw"
        if not raw_dir.is_dir():
            skip_count += 1
            continue

        for pos, recipe_profile in repeat.profiles.items():
            # Find corresponding Raw TIFF
            recipe_path = Path(recipe_profile.file_path)
            raw_path = raw_dir / recipe_path.name
            if not raw_path.exists():
                continue

            total_count += 1
            try:
                raw_profile = read_profile(raw_path)

                # Apply Order-2 flatten to both
                recipe_flat = polynomial_flatten(recipe_profile.z_nm, order=2)
                raw_flat = polynomial_flatten(raw_profile.z_nm, order=2)

                recipe_range = float(recipe_flat.max() - recipe_flat.min())
                raw_range = float(raw_flat.max() - raw_flat.min())
                diff = abs(recipe_range - raw_range)

                if diff <= _RANGE_TOL:
                    details.append({
                        "repeat": repeat.repeat_no,
                        "folder": repeat.directory.name,
                        "position": pos,
                        "recipe_range": f"{recipe_range:.3f}",
                        "raw_range": f"{raw_range:.3f}",
                        "diff_nm": f"{diff:.4f}",
                        "status": "PASS",
                    })
                else:
                    fail_count += 1
                    details.append({
                        "repeat": repeat.repeat_no,
                        "folder": repeat.directory.name,
                        "position": pos,
                        "recipe_range": f"{recipe_range:.3f}",
                        "raw_range": f"{raw_range:.3f}",
                        "diff_nm": f"{diff:.4f}",
                        "status": "FAIL",
                    })

            except Exception as e:
                fail_count += 1
                details.append({
                    "repeat": repeat.repeat_no,
                    "folder": repeat.directory.name,
                    "position": pos,
                    "recipe_range": "-",
                    "raw_range": "-",
                    "diff_nm": "-",
                    "status": f"ERROR: {e}",
                })

    if skip_count > 0 and total_count == 0:
        status = "WARN"
        summary = "Debug/Raw directories not found - skipped"
    elif fail_count > 0:
        status = "FAIL"
        summary = f"{fail_count}/{total_count} pairs differ after flatten"
    else:
        status = "PASS"
        summary = f"All {total_count} pairs equivalent after Order-2 flatten"

    return QCItemResult(
        check_id="QC-2",
        name="Data Equivalence",
        status=status,
        summary=summary,
        details=details,
    )


# ---------------------------------------------------------------------------
# QC-3: Scan Parameter Consistency
# ---------------------------------------------------------------------------

def _check_scan_parameters(recipe: RecipeData) -> QCItemResult:
    """Check scan parameters are consistent across all profiles."""
    param_names = [
        ("z_sensitivity_m", "Z Sensitivity (m)"),
        ("scan_size_um", "Scan Size (um)"),
        ("scan_speed_mm_s", "Scan Speed (mm/s)"),
        ("set_point", "Set Point"),
        ("z_servo_gain", "Z Servo Gain"),
    ]

    # Collect parameter values across all profiles
    param_values: dict[str, list[float]] = {attr: [] for attr, _ in param_names}
    profile_count = 0

    for repeat in recipe.repeats:
        for pos, profile in repeat.profiles.items():
            profile_count += 1
            for attr, _ in param_names:
                param_values[attr].append(getattr(profile, attr))

    if profile_count == 0:
        return QCItemResult(
            check_id="QC-3",
            name="Scan Parameters",
            status="WARN",
            summary="No profiles loaded",
            details=[],
        )

    details = []
    fail_count = 0

    for attr, display_name in param_names:
        values = np.array(param_values[attr])
        val_min = float(values.min())
        val_max = float(values.max())
        val_mean = float(values.mean())

        # Check consistency: all values should be identical (within floating-point tolerance)
        if val_mean != 0:
            deviation_pct = (val_max - val_min) / abs(val_mean) * 100
        else:
            deviation_pct = 0.0 if val_max == val_min else 100.0

        is_consistent = np.allclose(values, values[0], rtol=1e-9, atol=0)
        status = "PASS" if is_consistent else "FAIL"
        if not is_consistent:
            fail_count += 1

        details.append({
            "parameter": display_name,
            "expected": f"{values[0]:.6g}",
            "min": f"{val_min:.6g}",
            "max": f"{val_max:.6g}",
            "deviation": f"{deviation_pct:.4f}%",
            "status": status,
        })

    if fail_count > 0:
        status = "FAIL"
        summary = f"{fail_count}/{len(param_names)} parameters inconsistent"
    else:
        status = "PASS"
        summary = f"All {len(param_names)} parameters consistent ({profile_count} profiles)"

    return QCItemResult(
        check_id="QC-3",
        name="Scan Parameters",
        status=status,
        summary=summary,
        details=details,
    )


# ---------------------------------------------------------------------------
# QC-4: Position Completeness
# ---------------------------------------------------------------------------

def _check_position_completeness(recipe: RecipeData) -> QCItemResult:
    """Check all 9 positions are complete across all repeats."""
    details = []
    fail_count = 0
    total_count = 0

    for repeat in recipe.repeats:
        # Build lookup: position -> PointInfo
        point_by_pos = {}
        for pt in repeat.points:
            if pt.is_valid or pt.point_no >= 2:  # Exclude stabilization point
                point_by_pos[pt.position] = pt

        for pos in POSITION_LABELS:
            total_count += 1
            pt = point_by_pos.get(pos)
            has_profile = pos in repeat.profiles

            if pt is None:
                status = "FAIL"
                state = "MISSING"
                fail_count += 1
            elif pt.state != "COMPLETED":
                status = "FAIL"
                state = pt.state
                fail_count += 1
            elif not has_profile:
                status = "FAIL"
                state = "NO TIFF"
                fail_count += 1
            else:
                status = "PASS"
                state = "COMPLETED"

            details.append({
                "repeat": repeat.repeat_no,
                "folder": repeat.directory.name,
                "position": pos,
                "state": state,
                "has_profile": has_profile,
                "status": status,
            })

    if fail_count > 0:
        status = "FAIL"
        summary = f"{fail_count}/{total_count} position-repeat entries incomplete"
    else:
        status = "PASS"
        n_repeats = len(recipe.repeats)
        summary = f"All {len(POSITION_LABELS)} positions x {n_repeats} repeats complete"

    return QCItemResult(
        check_id="QC-4",
        name="Position Completeness",
        status=status,
        summary=summary,
        details=details,
    )


# ---------------------------------------------------------------------------
# QC-5: Statistical Outlier Detection
# ---------------------------------------------------------------------------

def _check_statistical_outliers(recipe: RecipeData) -> QCItemResult:
    """Detect outlier measurements using Median +/- 3*MAD criterion."""
    n_repeats = len(recipe.repeats)

    if n_repeats < 3:
        return QCItemResult(
            check_id="QC-5",
            name="Outlier Detection",
            status="PASS",
            summary=f"Skipped — insufficient repeats ({n_repeats} < 3)",
            details=[],
        )

    details = []
    outlier_count = 0

    for pos in POSITION_LABELS:
        # Collect OPM values for this position across repeats
        opm_values = []
        repeat_nos = []
        folders = []
        for repeat in recipe.repeats:
            if pos in repeat.profiles:
                opm_values.append(repeat.profiles[pos].opm_nm)
                repeat_nos.append(repeat.repeat_no)
                folders.append(repeat.directory.name)

        if len(opm_values) < 3:
            continue

        arr = np.array(opm_values)
        median_val = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median_val)))
        threshold = 3.0 * _MAD_SCALE * mad

        for i, (opm, rep_no, folder) in enumerate(zip(opm_values, repeat_nos, folders)):
            is_outlier = abs(opm - median_val) > threshold if threshold > 0 else False
            if is_outlier:
                outlier_count += 1

            details.append({
                "repeat": rep_no,
                "folder": folder,
                "position": pos,
                "opm_nm": f"{opm:.3f}",
                "median": f"{median_val:.3f}",
                "mad": f"{mad:.3f}",
                "threshold": f"{threshold:.3f}",
                "is_outlier": is_outlier,
                "status": "WARN" if is_outlier else "PASS",
            })

    if outlier_count > 0:
        status = "WARN"
        summary = f"{outlier_count} outlier(s) detected"
    else:
        status = "PASS"
        summary = "No outliers detected"

    return QCItemResult(
        check_id="QC-5",
        name="Outlier Detection",
        status=status,
        summary=summary,
        details=details,
    )


# ---------------------------------------------------------------------------
# QC-6: Pixel Count Consistency
# ---------------------------------------------------------------------------

def _check_pixel_count(recipe: RecipeData) -> QCItemResult:
    """Check all profiles have the expected pixel count (8192)."""
    details = []
    fail_count = 0
    total_count = 0

    for repeat in recipe.repeats:
        for pos in POSITION_LABELS:
            if pos not in repeat.profiles:
                continue
            total_count += 1
            profile = repeat.profiles[pos]
            px_count = len(profile.raw_data)

            if px_count != _EXPECTED_PIXELS:
                status = "FAIL"
                fail_count += 1
            else:
                status = "PASS"

            details.append({
                "repeat": repeat.repeat_no,
                "folder": repeat.directory.name,
                "position": pos,
                "pixel_count": px_count,
                "expected": _EXPECTED_PIXELS,
                "status": status,
            })

    if total_count == 0:
        return QCItemResult(
            check_id="QC-6",
            name="Pixel Count",
            status="WARN",
            summary="No profiles loaded",
            details=[],
        )

    if fail_count > 0:
        status = "FAIL"
        summary = f"{fail_count}/{total_count} profiles have non-standard pixel count"
    else:
        status = "PASS"
        summary = f"All {total_count} profiles have {_EXPECTED_PIXELS} pixels"

    return QCItemResult(
        check_id="QC-6",
        name="Pixel Count",
        status=status,
        summary=summary,
        details=details,
    )
