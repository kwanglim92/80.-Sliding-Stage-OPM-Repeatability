"""Position Detail Dialog — interactive deep-dive into a single position.

Double-click any subplot in the 3×3 Profile Charts grid to open this dialog.
Uses pyqtgraph for scroll-zoom, drag-pan, and real-time crosshair.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QFont, QColor, QPen
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QHeaderView, QSizePolicy,
)

from ..core.data_loader import RecipeData
from ..core.analyzer import AnalysisResult


# ── pyqtgraph global dark theme ──────────────────────────────
pg.setConfigOptions(antialias=True, background="#1e1e2e", foreground="#cdd6f4")

# Colors
_COLORS = {
    "bg": "#1e1e2e",
    "fg": "#cdd6f4",
    "grid": "#45475a",
    "accent": "#89b4fa",
    "mean": "#f5c2e7",
    "sigma": "#cba6f7",
    "overlay": [
        "#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8",
        "#fab387", "#cba6f7", "#f5c2e7", "#89dceb",
        "#b4befe", "#f5e0dc", "#eba0ac", "#a6d189",
        "#e78284", "#ef9f76", "#81c8be", "#ca9ee6",
    ],
}

DARK_STYLE = """
QDialog { background-color: #1e1e2e; }
QLabel { color: #cdd6f4; }
QCheckBox { color: #cdd6f4; font-size: 12px; spacing: 4px; }
QCheckBox::indicator { width: 14px; height: 14px; }
QCheckBox::indicator:checked {
    background-color: #89b4fa; border: 2px solid #b4befe; border-radius: 3px; }
QCheckBox::indicator:unchecked {
    background-color: #313244; border: 2px solid #585b70; border-radius: 3px; }
QGroupBox {
    border: 1px solid #45475a; border-radius: 6px;
    margin-top: 8px; padding-top: 16px;
    font-weight: bold; color: #89b4fa; font-size: 12px;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QTableWidget {
    background-color: #181825; color: #cdd6f4;
    gridline-color: #313244; border: 1px solid #45475a; font-size: 12px;
}
QTableWidget::item { padding: 4px; }
QHeaderView::section {
    background-color: #313244; color: #89b4fa;
    padding: 6px; border: 1px solid #45475a; font-weight: bold; font-size: 12px;
}
"""


class PositionDetailDialog(QDialog):
    """Interactive detail view for a single wafer position (pyqtgraph)."""

    def __init__(self, position: str, recipe: RecipeData,
                 result: Optional[AnalysisResult] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.position = position
        self.recipe = recipe
        self.result = result

        # Collect profiles for this position
        self.profiles: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        for repeat in recipe.repeats:
            if position in repeat.profiles:
                prof = repeat.profiles[position]
                self.profiles.append((
                    repeat.repeat_no,
                    prof.x_mm.astype(np.float64),
                    prof.z_nm.astype(np.float64),
                    float(prof.opm_nm),
                ))

        self.setWindowTitle(f"Position Detail — {position}")
        self.setMinimumSize(1000, 650)
        self.resize(1150, 720)
        self.setStyleSheet(DARK_STYLE)

        self._setup_ui()
        self._draw_chart()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ─── Left: pyqtgraph PlotWidget ───
        chart_container = QVBoxLayout()
        chart_container.setSpacing(2)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_widget.setLabel("bottom", "mm", **{"font-size": "11px"})
        self.plot_widget.setLabel("left", "nm", **{"font-size": "11px"})
        self.plot_widget.setTitle(
            f"{self.position} — {self.recipe.range_label}",
            size="14px", color=_COLORS["fg"])
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.getPlotItem().getViewBox().setMouseMode(
            pg.ViewBox.RectMode)  # default: drag to zoom box

        # Legend
        self.legend = self.plot_widget.addLegend(
            offset=(10, 10), labelTextSize="9px",
            brush=pg.mkBrush("#313244CC"), pen=pg.mkPen("#45475a"))

        # Crosshair
        self.vline = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen("#585b70", width=1, style=Qt.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False,
                                     pen=pg.mkPen("#585b70", width=1, style=Qt.DashLine))
        self.plot_widget.addItem(self.vline, ignoreBounds=True)
        self.plot_widget.addItem(self.hline, ignoreBounds=True)
        self.crosshair_label = QLabel("X: — mm  Y: — nm")
        self.crosshair_label.setStyleSheet(
            "font-size: 11px; color: #a6adc8; padding: 2px 6px;"
            "background-color: #181825; border-radius: 3px;")
        self.crosshair_label.setFixedHeight(22)
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        chart_container.addWidget(self.plot_widget)
        chart_container.addWidget(self.crosshair_label)
        main_layout.addLayout(chart_container, 7)

        # ─── Right: Controls + Stats ───
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)

        # Title
        title = QLabel(f"{self.position} — {self.recipe.range_label}")
        title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #89b4fa; padding: 4px;")
        right_panel.addWidget(title)

        subtitle = QLabel(f"{len(self.profiles)} Repeats")
        subtitle.setStyleSheet("font-size: 12px; color: #a6adc8; padding: 0 4px;")
        right_panel.addWidget(subtitle)

        # Interaction hint
        hint = QLabel("Scroll: Zoom  |  Drag: Pan  |  Right-click: Menu")
        hint.setStyleSheet(
            "font-size: 10px; color: #585b70; padding: 2px 4px; font-style: italic;")
        right_panel.addWidget(hint)

        # ─── Repeat Checkboxes ───
        repeat_group = QGroupBox("Repeat Toggle")
        repeat_layout = QGridLayout(repeat_group)
        repeat_layout.setSpacing(4)

        self.repeat_checkboxes: list[QCheckBox] = []
        for i, (rep_no, _, _, opm) in enumerate(self.profiles):
            cb = QCheckBox(f"R{rep_no}")
            cb.setChecked(True)
            color = _COLORS["overlay"][i % len(_COLORS["overlay"])]
            cb.setStyleSheet(
                f"QCheckBox {{ color: {color}; font-weight: bold; }}"
                f"QCheckBox::indicator:checked {{ background-color: {color};"
                f"border: 2px solid {color}; border-radius: 3px; }}")
            cb.toggled.connect(self._draw_chart)
            repeat_layout.addWidget(cb, i // 3, i % 3)
            self.repeat_checkboxes.append(cb)

        right_panel.addWidget(repeat_group)

        # ─── Overlay Options ───
        overlay_group = QGroupBox("Overlay Options")
        overlay_layout = QVBoxLayout(overlay_group)

        self.mean_cb = QCheckBox("Mean Profile")
        self.mean_cb.setChecked(True)
        self.mean_cb.setStyleSheet(
            f"QCheckBox {{ color: {_COLORS['mean']}; font-weight: bold; }}")
        self.mean_cb.toggled.connect(self._draw_chart)
        overlay_layout.addWidget(self.mean_cb)

        self.sigma_cb = QCheckBox("±1σ Band")
        self.sigma_cb.setChecked(False)
        self.sigma_cb.setStyleSheet(
            f"QCheckBox {{ color: {_COLORS['sigma']}; font-weight: bold; }}")
        self.sigma_cb.toggled.connect(self._draw_chart)
        overlay_layout.addWidget(self.sigma_cb)

        right_panel.addWidget(overlay_group)

        # ─── Stats Panel ───
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)

        # Per-repeat OPM table
        self.opm_table = QTableWidget()
        self.opm_table.setColumnCount(2)
        self.opm_table.setHorizontalHeaderLabels(["Repeat", "OPM (nm)"])
        self.opm_table.setRowCount(len(self.profiles))
        for i, (rep_no, _, _, opm) in enumerate(self.profiles):
            r_item = QTableWidgetItem(f"R{rep_no}")
            r_item.setTextAlignment(Qt.AlignCenter)
            self.opm_table.setItem(i, 0, r_item)
            v_item = QTableWidgetItem(f"{opm:.3f}")
            v_item.setTextAlignment(Qt.AlignCenter)
            self.opm_table.setItem(i, 1, v_item)
        self.opm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.opm_table.verticalHeader().setVisible(False)
        self.opm_table.setMaximumHeight(min(30 * len(self.profiles) + 30, 200))
        stats_layout.addWidget(self.opm_table)

        # Summary
        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.summary_label.setStyleSheet(
            "font-size: 12px; color: #cdd6f4; padding: 6px;"
            "background-color: #181825; border-radius: 4px;")
        self.summary_label.setWordWrap(True)
        self._update_summary()
        stats_layout.addWidget(self.summary_label)

        right_panel.addWidget(stats_group, 1)
        main_layout.addLayout(right_panel, 3)

    # ─── Crosshair ────────────────────────────────────────────

    def _on_mouse_moved(self, pos):
        """Update crosshair and coordinate label."""
        vb = self.plot_widget.getPlotItem().vb
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            self.vline.setPos(x)
            self.hline.setPos(y)
            self.crosshair_label.setText(f"X: {x:.3f} mm   Y: {y:.3f} nm")

    # ─── Stats ────────────────────────────────────────────────

    def _update_summary(self):
        if not self.profiles:
            self.summary_label.setText("No data")
            return

        opm_values = np.array([p[3] for p in self.profiles])
        z_arrays = [p[2] for p in self.profiles]

        if len(z_arrays) >= 2:
            stack = np.array(z_arrays, dtype=np.float64)
            pixel_range = stack.max(axis=0) - stack.min(axis=0)
            rep_max = float(pixel_range.max())
            rep_1sigma = float(pixel_range.std(ddof=0))
        else:
            rep_max = rep_1sigma = 0.0

        mean_z = np.mean(z_arrays, axis=0) if z_arrays else np.array([])
        mean_val = float(mean_z.mean()) if len(mean_z) > 0 else 0.0
        std_val = float(np.std([z.mean() for z in z_arrays])) if z_arrays else 0.0

        self.summary_label.setText(
            f"<b>Rep. Max:</b> {rep_max:.3f} nm<br>"
            f"<b>Rep. 1\u03c3:</b> {rep_1sigma:.3f} nm<br>"
            f"<b>OPM Max:</b> {float(opm_values.max()):.3f} nm<br>"
            f"<b>OPM Mean:</b> {float(opm_values.mean()):.3f} nm<br>"
            f"<b>OPM Stdev:</b> {float(opm_values.std(ddof=0)):.3f} nm<br>"
            f"<br>"
            f"<b>Z Mean:</b> {mean_val:.2f} nm<br>"
            f"<b>Z Spread:</b> \u00b1{std_val:.2f} nm"
        )

    # ─── Chart Drawing ────────────────────────────────────────

    def _draw_chart(self):
        """Redraw all curves based on checkbox states."""
        self.plot_widget.clear()
        # Re-add crosshair after clear
        self.plot_widget.addItem(self.vline, ignoreBounds=True)
        self.plot_widget.addItem(self.hline, ignoreBounds=True)

        # Re-add legend
        if self.legend is not None:
            try:
                self.legend.scene().removeItem(self.legend)
            except Exception:
                pass
        self.legend = self.plot_widget.addLegend(
            offset=(10, 10), labelTextSize="9px",
            brush=pg.mkBrush("#313244CC"), pen=pg.mkPen("#45475a"))

        visible_z = []
        visible_x = None

        for i, (rep_no, x_mm, z_nm, opm) in enumerate(self.profiles):
            if i < len(self.repeat_checkboxes) and self.repeat_checkboxes[i].isChecked():
                color = _COLORS["overlay"][i % len(_COLORS["overlay"])]
                pen = pg.mkPen(color, width=1.2)
                self.plot_widget.plot(
                    x_mm, z_nm, pen=pen,
                    name=f"R{rep_no} (OPM {opm:.1f})")
                visible_z.append(z_nm)
                visible_x = x_mm

        # Mean profile
        if visible_z and self.mean_cb.isChecked() and visible_x is not None:
            mean_z = np.mean(visible_z, axis=0)
            pen = pg.mkPen(_COLORS["mean"], width=2.5)
            self.plot_widget.plot(visible_x, mean_z, pen=pen, name="Mean")

            # ±1σ band
            if self.sigma_cb.isChecked() and len(visible_z) >= 2:
                std_z = np.std(visible_z, axis=0, ddof=0)
                upper = mean_z + std_z
                lower = mean_z - std_z
                fill = pg.FillBetweenItem(
                    pg.PlotDataItem(visible_x, upper),
                    pg.PlotDataItem(visible_x, lower),
                    brush=pg.mkBrush(QColor(_COLORS["sigma"]).lighter(120).name() + "30"))
                self.plot_widget.addItem(fill)
