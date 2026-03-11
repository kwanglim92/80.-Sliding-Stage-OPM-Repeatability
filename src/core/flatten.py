"""Flatten (polynomial regression detrending) for profile data.

Implements 0th~12th order polynomial regression fitting to remove
surface tilt, curvature, and waviness from profile data.

Features:
    - Regression Order 0~12 (0차=mean subtraction, 1=linear, 2=quadratic, ...)
    - Edge Pixel exclusion (default 1% each side, AFP-compatible)
    - Outlier removal (percentile or pixel count based)
    - Undo/Redo history stack
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FlattenResult:
    """Result of a flatten operation."""
    original: np.ndarray        # Original Z data (nm)
    flattened: np.ndarray       # Flattened Z data (nm)
    regression: np.ndarray      # Regression curve (nm)
    residual: np.ndarray        # Residual = original - regression
    order: int                  # Polynomial order used
    coefficients: np.ndarray    # Polynomial coefficients
    edge_percent: float         # Edge exclusion percentage
    opm_before: float           # OPM before flatten
    opm_after: float            # OPM after flatten
    rms_before: float           # RMS before flatten
    rms_after: float            # RMS after flatten


class FlattenProcessor:
    """Profile data flatten processor with undo/redo support."""

    def __init__(self, max_history: int = 20):
        self._history: list[FlattenResult] = []
        self._redo_stack: list[FlattenResult] = []
        self._max_history = max_history

    @property
    def can_undo(self) -> bool:
        return len(self._history) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def last_result(self) -> Optional[FlattenResult]:
        return self._history[-1] if self._history else None

    def flatten(self, z_data: np.ndarray, x_data: Optional[np.ndarray] = None,
                order: int = 1, edge_percent: float = 1.0,
                outlier_mode: str = "none", outlier_value: float = 0.0) -> FlattenResult:
        """Apply polynomial flatten to profile data.

        Args:
            z_data: Z values in nm (1D array).
            x_data: X values (optional, auto-generated if None).
            order: Polynomial regression order (0~12).
            edge_percent: Percentage of edge pixels to exclude (each side, default 1%).
            outlier_mode: "none", "percentile", or "pixels".
            outlier_value: Percentile threshold (0~100) or pixel count for outlier removal.

        Returns:
            FlattenResult with original, flattened, and regression data.
        """
        if order < 0 or order > 12:
            raise ValueError(f"Regression order must be 0~12, got {order}")

        n = len(z_data)
        if x_data is None:
            x_data = np.arange(n, dtype=np.float64)

        # --- Edge exclusion ---
        edge_pixels = max(1, int(n * edge_percent / 100.0))
        inner_start = edge_pixels
        inner_end = n - edge_pixels

        x_inner = x_data[inner_start:inner_end]
        z_inner = z_data[inner_start:inner_end].copy()

        # --- Outlier removal (optional) ---
        mask = np.ones(len(z_inner), dtype=bool)
        if outlier_mode == "percentile" and outlier_value > 0:
            # Remove pixels with deviation above percentile threshold
            mean_val = np.mean(z_inner)
            deviation = np.abs(z_inner - mean_val)
            threshold = np.percentile(deviation, 100 - outlier_value)
            mask = deviation <= threshold
        elif outlier_mode == "pixels" and outlier_value > 0:
            # Remove top N pixels by deviation
            mean_val = np.mean(z_inner)
            deviation = np.abs(z_inner - mean_val)
            n_remove = min(int(outlier_value), len(z_inner) - order - 1)
            if n_remove > 0:
                threshold_idx = np.argsort(deviation)[-n_remove]
                threshold = deviation[threshold_idx]
                mask = deviation < threshold

        x_fit = x_inner[mask]
        z_fit = z_inner[mask]

        # --- Polynomial fitting ---
        if order == 0:
            # 0th order = mean subtraction
            coefficients = np.array([np.mean(z_fit)])
            regression_full = np.full(n, coefficients[0])
        else:
            # Normalize x for numerical stability
            x_mean = x_data.mean()
            x_std = x_data.std()
            if x_std == 0:
                x_std = 1.0

            x_norm = (x_data - x_mean) / x_std
            x_fit_norm = (x_fit - x_mean) / x_std

            coefficients = np.polyfit(x_fit_norm, z_fit, order)
            regression_full = np.polyval(coefficients, x_norm)

        # --- Apply flatten ---
        flattened = z_data - regression_full

        # --- Statistics ---
        opm_before = float(z_data.max() - z_data.min())
        opm_after = float(flattened.max() - flattened.min())
        rms_before = float(np.sqrt(np.mean((z_data - z_data.mean()) ** 2)))
        rms_after = float(np.sqrt(np.mean(flattened ** 2)))

        result = FlattenResult(
            original=z_data.copy(),
            flattened=flattened,
            regression=regression_full,
            residual=flattened,  # same as z_data - regression
            order=order,
            coefficients=coefficients,
            edge_percent=edge_percent,
            opm_before=opm_before,
            opm_after=opm_after,
            rms_before=rms_before,
            rms_after=rms_after,
        )

        # Update history
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._redo_stack.clear()

        return result

    def undo(self) -> Optional[FlattenResult]:
        """Undo last flatten operation. Returns previous result or None."""
        if not self._history:
            return None
        result = self._history.pop()
        self._redo_stack.append(result)
        return self._history[-1] if self._history else None

    def redo(self) -> Optional[FlattenResult]:
        """Redo previously undone flatten. Returns re-applied result or None."""
        if not self._redo_stack:
            return None
        result = self._redo_stack.pop()
        self._history.append(result)
        return result

    def clear_history(self) -> None:
        """Clear all undo/redo history."""
        self._history.clear()
        self._redo_stack.clear()


def quick_flatten(z_data: np.ndarray, order: int = 1,
                  edge_percent: float = 1.0) -> np.ndarray:
    """Quick flatten without history tracking.

    Args:
        z_data: Z values in nm.
        order: Polynomial order (0~12).
        edge_percent: Edge exclusion percentage.

    Returns:
        Flattened Z values.
    """
    proc = FlattenProcessor(max_history=1)
    result = proc.flatten(z_data, order=order, edge_percent=edge_percent)
    return result.flattened
