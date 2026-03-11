"""Data loader for Sliding Stage OPM Repeatability measurement data.

Automatically parses directory structure, Info CSV files, and loads
TIFF profile data with proper Point→Position mapping.

Directory Structure:
    data/{range: 25mm|10mm|5mm|1mm}/
        {Sample|Lot}{N}/                  ← One Repeat
            {prefix}.csv                  ← Measurement metadata
            {prefix}_NNNN_Height.tiff     ← Height profile data
            {prefix}_NNNN_Z Drive.tiff    ← Z Drive profile data
            Info/{prefix}_Info.csv        ← Point→Position mapping

Wafer 9-Point Grid (3×3):
    LT  CT  RT     (Top row,    Y ≈ +89mm)
    LM  CM  RM     (Middle row, Y ≈ +9mm)
    LB  CB  RB     (Bottom row, Y ≈ -71mm)
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .tiff_reader import ProfileData, read_profile

# 9-point wafer position labels (in measurement order)
POSITION_LABELS = ["1_LT", "2_CT", "3_RT", "4_LM", "5_CM", "6_RM", "7_LB", "8_CB", "9_RB"]

# Mapping from position label to grid (row, col)
POSITION_GRID = {
    "1_LT": (0, 0), "2_CT": (0, 1), "3_RT": (0, 2),
    "4_LM": (1, 0), "5_CM": (1, 1), "6_RM": (1, 2),
    "7_LB": (2, 0), "8_CB": (2, 1), "9_RB": (2, 2),
}


@dataclass
class PointInfo:
    """Information about a single measurement point."""
    point_no: int           # 1-based point number
    x_um: float             # X coordinate in µm
    y_um: float             # Y coordinate in µm
    method_id: str          # Method ID (e.g., "25mm_1V")
    state: str              # COMPLETED, FAILED, etc.
    filename: str           # Base filename (without extension)
    date: str               # Measurement date/time
    position: str           # Position label (e.g., "1_LT")
    is_valid: bool          # Whether this point is valid for analysis


@dataclass
class RepeatData:
    """Data for a single repeat (one sample directory)."""
    repeat_no: int                  # 1-based repeat number
    directory: Path                 # Sample/Lot directory path
    lot_id: str                     # Lot ID from metadata CSV
    sample_id: str                  # Sample ID from metadata CSV
    recipe_id: str                  # Recipe ID
    points: list[PointInfo]         # All measurement points
    profiles: dict[str, ProfileData] = field(default_factory=dict, repr=False)
    # profiles keyed by position label, e.g., "1_LT"

    @property
    def valid_points(self) -> list[PointInfo]:
        """Points valid for analysis (excluding stabilization point)."""
        return [p for p in self.points if p.is_valid]


@dataclass
class RecipeData:
    """Data for a single recipe range (e.g., 25mm)."""
    range_mm: int                   # Scan range in mm (25, 10, 5, 1)
    range_label: str                # "25mm", "10mm", etc.
    directory: Path                 # Recipe directory path
    repeats: list[RepeatData]       # All repeats, sorted by repeat_no

    @property
    def repeat_count(self) -> int:
        return len(self.repeats)

    @property
    def position_labels(self) -> list[str]:
        """Available position labels across repeats."""
        if not self.repeats:
            return []
        return [p.position for p in self.repeats[0].valid_points]


@dataclass
class DataSet:
    """Complete dataset across all recipes."""
    root_directory: Path
    recipes: dict[str, RecipeData]  # keyed by range_label ("25mm", etc.)

    @property
    def available_ranges(self) -> list[str]:
        return sorted(self.recipes.keys(), key=lambda x: int(x.replace("mm", "")), reverse=True)


def _parse_info_csv(info_path: Path) -> list[PointInfo]:
    """Parse Info CSV to get point→position mapping.

    The Info CSV has a header section followed by point data:
        Point No, X (um), Y (um), Method ID, State, FileName, Date
    """
    points = []

    with open(info_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Find the header row with "Point No"
    data_start = -1
    for i, line in enumerate(lines):
        if "Point No" in line:
            data_start = i + 1
            break

    if data_start < 0:
        return points

    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue

        point_no = int(parts[0].strip())
        x_um = float(parts[1].strip())
        y_um = float(parts[2].strip())
        method_id = parts[3].strip()
        state = parts[4].strip()
        filename = parts[5].strip()
        date = parts[6].strip() if len(parts) > 6 else ""

        # Determine position label from point number
        # Point 1 & 2 share same XY (Point 1 = stabilization)
        # Points map to: 1→1_LT(stab), 2→1_LT, 3→2_CT, 4→3_RT, ...
        if point_no == 1:
            position = "1_LT"
            is_valid = False  # Stabilization point
        elif 2 <= point_no <= 10:
            position = POSITION_LABELS[point_no - 2]
            is_valid = state == "COMPLETED"
        else:
            position = f"P{point_no}"
            is_valid = state == "COMPLETED"

        points.append(PointInfo(
            point_no=point_no,
            x_um=x_um,
            y_um=y_um,
            method_id=method_id,
            state=state,
            filename=filename,
            date=date,
            position=position,
            is_valid=is_valid,
        ))

    return points


def _parse_metadata_csv(csv_path: Path) -> dict[str, str]:
    """Parse the measurement metadata CSV (key-value pairs)."""
    metadata = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            key, _, value = line.partition(",")
            metadata[key.strip()] = value.strip()
    return metadata


def _detect_range_mm(directory_name: str) -> Optional[int]:
    """Detect recipe range from directory name (e.g., '25mm' → 25)."""
    match = re.match(r"(\d+)mm", directory_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _find_repeat_directories(recipe_dir: Path) -> list[Path]:
    """Find and sort sample/lot directories within a recipe directory."""
    dirs = []
    for d in sorted(recipe_dir.iterdir()):
        if not d.is_dir():
            continue
        # Skip Capture and other non-data directories
        if d.name.lower() in ("capture", "debug", "info", "log"):
            continue
        # Check if it contains TIFF files or Info subdirectory
        has_tiff = any(d.glob("*_Height.tiff"))
        has_info = (d / "Info").is_dir()
        if has_tiff or has_info:
            dirs.append(d)
    return dirs


def _load_repeat(repeat_dir: Path, repeat_no: int,
                 signal_source: str = "Height") -> RepeatData:
    """Load a single repeat's data from a sample/lot directory.

    Args:
        repeat_dir: Path to the sample/lot directory.
        repeat_no: 1-based repeat number.
        signal_source: "Height" or "Z Drive".
    """
    # Parse metadata
    csv_files = list(repeat_dir.glob("*.csv"))
    metadata = {}
    if csv_files:
        metadata = _parse_metadata_csv(csv_files[0])

    # Parse Info CSV for point mapping
    info_dir = repeat_dir / "Info"
    points = []
    if info_dir.is_dir():
        info_files = list(info_dir.glob("*_Info.csv"))
        if info_files:
            points = _parse_info_csv(info_files[0])

    # If no Info CSV found, try to build from TIFF filenames
    if not points:
        suffix = f"_{signal_source}.tiff"
        tiff_files = sorted(repeat_dir.glob(f"*{suffix}"))
        for i, tf in enumerate(tiff_files):
            point_no = i + 1
            if point_no == 1:
                position = "1_LT"
                is_valid = False
            elif 2 <= point_no <= 10:
                position = POSITION_LABELS[point_no - 2]
                is_valid = True
            else:
                position = f"P{point_no}"
                is_valid = True

            points.append(PointInfo(
                point_no=point_no, x_um=0, y_um=0,
                method_id="", state="COMPLETED",
                filename=tf.stem.rsplit("_", 1)[0],
                date="", position=position, is_valid=is_valid,
            ))

    return RepeatData(
        repeat_no=repeat_no,
        directory=repeat_dir,
        lot_id=metadata.get("Lot ID", ""),
        sample_id=metadata.get("Sample ID", ""),
        recipe_id=metadata.get("Recipe ID", ""),
        points=points,
    )


def load_profiles_for_repeat(repeat: RepeatData,
                              signal_source: str = "Height") -> None:
    """Load TIFF profile data for all valid points in a repeat.

    Populates repeat.profiles dict keyed by position label.
    """
    suffix = f"_{signal_source}.tiff"

    for point in repeat.valid_points:
        # Find matching TIFF file
        tiff_pattern = f"{point.filename}_{signal_source}.tiff"
        tiff_files = list(repeat.directory.glob(tiff_pattern))

        if not tiff_files:
            # Try alternate pattern
            tiff_files = list(repeat.directory.glob(f"*_{point.point_no:04d}_{signal_source}.tiff"))

        if tiff_files:
            profile = read_profile(tiff_files[0])
            repeat.profiles[point.position] = profile


def load_recipe(recipe_dir: str | Path, signal_source: str = "Height",
                load_profiles: bool = True) -> RecipeData:
    """Load all repeat data for a single recipe.

    Args:
        recipe_dir: Path to recipe directory (e.g., data/25mm/).
        signal_source: "Height" or "Z Drive".
        load_profiles: If True, also load TIFF profile data.
    """
    recipe_dir = Path(recipe_dir)
    range_mm = _detect_range_mm(recipe_dir.name)
    range_label = recipe_dir.name

    repeat_dirs = _find_repeat_directories(recipe_dir)
    repeats = []

    for i, rdir in enumerate(repeat_dirs, start=1):
        repeat = _load_repeat(rdir, repeat_no=i, signal_source=signal_source)
        if load_profiles:
            load_profiles_for_repeat(repeat, signal_source=signal_source)
        repeats.append(repeat)

    return RecipeData(
        range_mm=range_mm or 0,
        range_label=range_label,
        directory=recipe_dir,
        repeats=repeats,
    )


def load_dataset(root_dir: str | Path, signal_source: str = "Height",
                 load_profiles: bool = True) -> DataSet:
    """Load complete dataset from root directory.

    Scans for recipe directories (25mm, 10mm, 5mm, 1mm) and loads all data.

    Args:
        root_dir: Path to data root directory containing recipe subdirs.
        signal_source: "Height" or "Z Drive".
        load_profiles: If True, load TIFF profile data (can be slow).
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    recipes = {}
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        range_mm = _detect_range_mm(subdir.name)
        if range_mm is not None:
            recipe = load_recipe(subdir, signal_source=signal_source,
                                 load_profiles=load_profiles)
            recipes[recipe.range_label] = recipe

    return DataSet(root_directory=root, recipes=recipes)
