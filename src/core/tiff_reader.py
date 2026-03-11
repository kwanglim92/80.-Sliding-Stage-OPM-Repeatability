"""Park Systems TIFF file reader for profile data.

Reads Park Systems custom TIFF format containing 1D profile scan data.
Supports both Height and Z Drive signal sources.

TIFF Structure:
    - Image: Rendered RGB chart (informational only)
    - Tag 50434: Raw profile data (8192 x float32) in DAC units
    - Tag 50435: Binary header (580 bytes) with scan parameters
    - Tag 50441: XML extended header with additional metadata

Z Conversion Formula:
    Z(nm) = raw_value / 2^20 * |Z_sensitivity| * 1e9
    where Z_sensitivity is in meters (from header offset 220)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:
    raise ImportError("tifffile is required: pip install tifffile")

# Park Systems TIFF custom tag IDs
TAG_RAW_DATA = 50434      # float32 profile data
TAG_HEADER = 50435        # binary header (580 bytes)
TAG_XML_HEADER = 50441    # XML extended header

# Binary header field offsets (byte positions)
_HDR_VERSION = 0           # int32: version
_HDR_SOURCE_NAME = 4       # UTF-16-LE: source channel name (32 chars)
_HDR_UNIT_NAME = 68        # UTF-16-LE: head mode / unit (32 chars)
_HDR_SCAN_SIZE = 140       # float64: scan size in µm
_HDR_XY_OFFSET_X = 156     # float64: XY offset X
_HDR_XY_OFFSET_Y = 164     # float64: XY offset Y
_HDR_SCAN_SPEED = 172      # float64: scan speed in mm/s
_HDR_SET_POINT = 180       # float64: set point
_HDR_Z_SENSITIVITY = 220   # float64: Z sensitivity in meters
_HDR_Z_SCALE = 228         # float64: Z scale factor
_HDR_Z_SERVO_GAIN = 284    # float64: Z servo gain

# DAC resolution
DAC_BITS = 20
DAC_FULL_SCALE = 2 ** DAC_BITS  # 1,048,576


@dataclass
class ProfileData:
    """Container for a single profile measurement."""

    x_mm: np.ndarray            # X axis in mm (0 to scan_size)
    z_nm: np.ndarray            # Z values in nm
    source: str                 # Signal source: "Height" or "Z Drive"
    scan_size_um: float         # Scan size in µm
    scan_speed_mm_s: float      # Scan speed in mm/s
    z_sensitivity_m: float      # Z sensitivity in meters
    set_point: float            # Set point value
    z_servo_gain: float         # Z servo gain
    head_mode: str              # Head mode (e.g., "C-AFM")
    file_path: str              # Original file path
    raw_data: np.ndarray = field(repr=False)  # Raw DAC values (preserved for re-conversion)

    @property
    def pixel_count(self) -> int:
        return len(self.z_nm)

    @property
    def scan_size_mm(self) -> float:
        return self.scan_size_um / 1000.0

    @property
    def opm_nm(self) -> float:
        """OPM = Max - Min of the profile (nm)."""
        return float(self.z_nm.max() - self.z_nm.min())

    @property
    def rms_nm(self) -> float:
        """RMS roughness of the profile (nm)."""
        mean = self.z_nm.mean()
        return float(np.sqrt(np.mean((self.z_nm - mean) ** 2)))

    @property
    def range_nm(self) -> float:
        """Alias for opm_nm."""
        return self.opm_nm


def _parse_header(header_bytes: bytes) -> dict:
    """Parse the 580-byte Park Systems binary header."""
    if len(header_bytes) < 236:
        raise ValueError(f"Header too short: {len(header_bytes)} bytes (expected ≥236)")

    def _read_utf16(data: bytes, offset: int, max_chars: int = 32) -> str:
        end = offset + max_chars * 2
        return data[offset:end].decode("utf-16-le", errors="replace").split("\x00")[0]

    def _read_f64(data: bytes, offset: int) -> float:
        return struct.unpack_from("<d", data, offset)[0]

    return {
        "version": struct.unpack_from("<I", header_bytes, _HDR_VERSION)[0],
        "source": _read_utf16(header_bytes, _HDR_SOURCE_NAME),
        "head_mode": _read_utf16(header_bytes, _HDR_UNIT_NAME),
        "scan_size_um": _read_f64(header_bytes, _HDR_SCAN_SIZE),
        "scan_speed_mm_s": _read_f64(header_bytes, _HDR_SCAN_SPEED),
        "set_point": _read_f64(header_bytes, _HDR_SET_POINT),
        "z_sensitivity_m": _read_f64(header_bytes, _HDR_Z_SENSITIVITY),
        "z_scale": _read_f64(header_bytes, _HDR_Z_SCALE),
        "z_servo_gain": _read_f64(header_bytes, _HDR_Z_SERVO_GAIN),
    }


def read_profile(file_path: str | Path) -> ProfileData:
    """Read a Park Systems TIFF profile file.

    Args:
        file_path: Path to .tiff file (Height or Z Drive).

    Returns:
        ProfileData with calibrated X(mm) and Z(nm) arrays.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is not recognized.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {file_path}")

    with tifffile.TiffFile(str(file_path)) as tif:
        page = tif.pages[0]

        # Extract raw profile data (Tag 50434)
        tag_data = page.tags.get(TAG_RAW_DATA)
        if tag_data is None:
            raise ValueError(f"No raw profile data (tag {TAG_RAW_DATA}) in: {file_path}")

        raw_bytes = tag_data.value
        if not isinstance(raw_bytes, bytes):
            raise ValueError(f"Unexpected data type for tag {TAG_RAW_DATA}: {type(raw_bytes)}")

        raw_data = np.frombuffer(raw_bytes, dtype="<f4").copy()

        # Extract header (Tag 50435)
        tag_header = page.tags.get(TAG_HEADER)
        if tag_header is None:
            raise ValueError(f"No header (tag {TAG_HEADER}) in: {file_path}")

        header = _parse_header(tag_header.value)

    # --- Calibrate ---
    # Z conversion: DAC units → nm
    z_sens_m = header["z_sensitivity_m"]
    z_nm = raw_data / DAC_FULL_SCALE * abs(z_sens_m) * 1e9

    # X axis: 0 to scan_size in mm
    scan_size_mm = header["scan_size_um"] / 1000.0
    x_mm = np.linspace(0, scan_size_mm, len(raw_data), endpoint=False)

    return ProfileData(
        x_mm=x_mm,
        z_nm=z_nm,
        source=header["source"],
        scan_size_um=header["scan_size_um"],
        scan_speed_mm_s=header["scan_speed_mm_s"],
        z_sensitivity_m=z_sens_m,
        set_point=header["set_point"],
        z_servo_gain=header["z_servo_gain"],
        head_mode=header["head_mode"],
        file_path=str(file_path),
        raw_data=raw_data,
    )


def detect_signal_source(file_path: str | Path) -> str:
    """Detect signal source from filename.

    Returns 'Height' or 'Z Drive' based on filename pattern.
    """
    name = Path(file_path).stem
    if "Z Drive" in name or "Z_Drive" in name:
        return "Z Drive"
    elif "Height" in name:
        return "Height"
    else:
        return "Unknown"
