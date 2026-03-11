"""Time analysis module for measurement duration tracking.

Extracts timestamps from Park Systems measurement data to compute:
- Per-repeat start/end time and duration
- Per-point measurement timestamps
- Inter-repeat gaps (continuity check)
- Recipe-level total duration
- Per-point average duration (for repeat count estimation)

Data Sources:
    1. Sample/Lot CSV: Start Time, End Time fields
    2. Info CSV:       Per-point Date column (ms precision)
    3. Log CSV:        Full job-level timing (tack time)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Timestamp formats used by Park Systems
_FMT_CSV = "%Y.%m.%d %H:%M:%S"          # Sample1.csv: "2025.12.30 21:16:16"
_FMT_INFO = "%Y/%m/%d %H:%M:%S.%f"       # Info CSV:    "2025/12/30 21:23:48.546"


def _parse_time(s: str) -> Optional[datetime]:
    """Parse a timestamp string, trying multiple formats."""
    s = s.strip()
    if not s:
        return None
    for fmt in [_FMT_INFO, _FMT_CSV, "%Y/%m/%d %H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _fmt_duration(td: timedelta) -> str:
    """Format timedelta as human readable string."""
    total_sec = int(td.total_seconds())
    if total_sec < 0:
        return "—"
    hours, rem = divmod(total_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    else:
        return f"{seconds}s"


@dataclass
class PointTiming:
    """Timing for a single measurement point."""
    point_no: int
    position: str               # e.g., "1_LT"
    timestamp: Optional[datetime]
    is_stabilization: bool      # Point 1 = stabilization


@dataclass
class RepeatTiming:
    """Timing for a single repeat."""
    repeat_no: int
    directory_name: str
    start_time: Optional[datetime]   # From metadata CSV
    end_time: Optional[datetime]     # From metadata CSV
    point_times: list[PointTiming] = field(default_factory=list)

    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def duration_str(self) -> str:
        d = self.duration
        return _fmt_duration(d) if d else "—"

    @property
    def first_point_time(self) -> Optional[datetime]:
        """Time of first valid (non-stabilization) point."""
        for pt in self.point_times:
            if not pt.is_stabilization and pt.timestamp:
                return pt.timestamp
        return None

    @property
    def last_point_time(self) -> Optional[datetime]:
        """Time of last point."""
        for pt in reversed(self.point_times):
            if pt.timestamp:
                return pt.timestamp
        return None

    @property
    def measurement_duration(self) -> Optional[timedelta]:
        """Duration of actual measurements (first to last point)."""
        ft = self.first_point_time
        lt = self.last_point_time
        if ft and lt:
            return lt - ft
        return None

    @property
    def per_point_duration_sec(self) -> Optional[float]:
        """Average seconds per point (excluding stabilization)."""
        valid = [p for p in self.point_times if not p.is_stabilization and p.timestamp]
        if len(valid) < 2:
            return None
        total = (valid[-1].timestamp - valid[0].timestamp).total_seconds()
        return total / (len(valid) - 1)


@dataclass
class RecipeTiming:
    """Timing for a complete recipe across all repeats."""
    range_label: str
    repeats: list[RepeatTiming]

    @property
    def total_duration(self) -> Optional[timedelta]:
        """Total duration from first repeat start to last repeat end."""
        starts = [r.start_time for r in self.repeats if r.start_time]
        ends = [r.end_time for r in self.repeats if r.end_time]
        if starts and ends:
            return max(ends) - min(starts)
        return None

    @property
    def total_duration_str(self) -> str:
        d = self.total_duration
        return _fmt_duration(d) if d else "—"

    @property
    def avg_repeat_duration(self) -> Optional[timedelta]:
        """Average repeat duration."""
        durations = [r.duration for r in self.repeats if r.duration]
        if durations:
            total = sum(d.total_seconds() for d in durations)
            return timedelta(seconds=total / len(durations))
        return None

    @property
    def avg_per_point_sec(self) -> Optional[float]:
        """Average seconds per measurement point across all repeats."""
        vals = [r.per_point_duration_sec for r in self.repeats
                if r.per_point_duration_sec]
        return sum(vals) / len(vals) if vals else None

    @property
    def gaps(self) -> list[dict]:
        """Gaps between consecutive repeats (for continuity analysis)."""
        result = []
        sorted_reps = sorted(
            [r for r in self.repeats if r.end_time and r.start_time],
            key=lambda r: r.start_time
        )
        for i in range(len(sorted_reps) - 1):
            curr = sorted_reps[i]
            nxt = sorted_reps[i + 1]
            gap = nxt.start_time - curr.end_time
            result.append({
                "from_repeat": curr.repeat_no,
                "to_repeat": nxt.repeat_no,
                "gap": gap,
                "gap_str": _fmt_duration(gap),
                "gap_sec": gap.total_seconds(),
                "is_continuous": gap.total_seconds() < 120,  # < 2min = continuous
            })
        return result

    @property
    def is_continuous(self) -> bool:
        """Whether all repeats were measured continuously (no large gaps)."""
        return all(g["is_continuous"] for g in self.gaps)

    def estimate_duration(self, repeat_count: int) -> str:
        """Estimate total measurement time for given repeat count."""
        avg = self.avg_repeat_duration
        if avg:
            estimated = avg * repeat_count
            return _fmt_duration(estimated)
        return "—"


def extract_repeat_timing(repeat_dir: Path, repeat_no: int) -> RepeatTiming:
    """Extract timing information from a single repeat directory."""
    start_time = None
    end_time = None

    # Source 1: Metadata CSV (Start Time, End Time)
    csv_files = list(repeat_dir.glob("*.csv"))
    for cf in csv_files:
        try:
            with open(cf, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Start Time,"):
                        start_time = _parse_time(line.split(",", 1)[1])
                    elif line.startswith("End Time,"):
                        end_time = _parse_time(line.split(",", 1)[1])
        except Exception:
            pass

    # Source 2: Info CSV (per-point timestamps)
    point_times = []
    info_dir = repeat_dir / "Info"
    if info_dir.is_dir():
        info_files = list(info_dir.glob("*_Info.csv"))
        if info_files:
            try:
                with open(info_files[0], "r", encoding="utf-8-sig") as f:
                    lines = f.readlines()

                data_start = -1
                for i, line in enumerate(lines):
                    if "Point No" in line:
                        data_start = i + 1
                        break

                if data_start >= 0:
                    from .data_loader import POSITION_LABELS
                    for line in lines[data_start:]:
                        parts = line.strip().split(",")
                        if len(parts) < 7:
                            continue
                        point_no = int(parts[0].strip())
                        date_str = parts[6].strip() if len(parts) > 6 else ""
                        ts = _parse_time(date_str)

                        if point_no == 1:
                            pos = "1_LT"
                            is_stab = True
                        elif 2 <= point_no <= 10:
                            pos = POSITION_LABELS[point_no - 2]
                            is_stab = False
                        else:
                            pos = f"P{point_no}"
                            is_stab = False

                        point_times.append(PointTiming(
                            point_no=point_no,
                            position=pos,
                            timestamp=ts,
                            is_stabilization=is_stab,
                        ))
            except Exception:
                pass

    return RepeatTiming(
        repeat_no=repeat_no,
        directory_name=repeat_dir.name,
        start_time=start_time,
        end_time=end_time,
        point_times=point_times,
    )


def extract_recipe_timing(recipe) -> RecipeTiming:
    """Extract timing for a complete recipe.

    Args:
        recipe: RecipeData object with loaded directory info.
    """
    repeat_timings = []
    for repeat in recipe.repeats:
        rt = extract_repeat_timing(repeat.directory, repeat.repeat_no)
        repeat_timings.append(rt)

    return RecipeTiming(
        range_label=recipe.range_label,
        repeats=repeat_timings,
    )


def format_timing_summary(timing: RecipeTiming) -> list[dict]:
    """Generate summary table rows for timing analysis."""
    rows = []

    for rt in timing.repeats:
        gap_info = ""
        for g in timing.gaps:
            if g["to_repeat"] == rt.repeat_no:
                gap_info = g["gap_str"]
                if not g["is_continuous"]:
                    gap_info += " ⚠"

        rows.append({
            "Repeat": f"R{rt.repeat_no}",
            "Folder": rt.directory_name,
            "Start": rt.start_time.strftime("%H:%M:%S") if rt.start_time else "—",
            "End": rt.end_time.strftime("%H:%M:%S") if rt.end_time else "—",
            "Duration": rt.duration_str,
            "Per Point": f"{rt.per_point_duration_sec:.0f}s" if rt.per_point_duration_sec else "—",
            "Gap": gap_info,
        })

    return rows
