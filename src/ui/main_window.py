"""PySide6 GUI for Sliding Stage OPM Repeatability Analyzer.

Main window with:
- Data loading panel (folder selection + tree view)
- Settings panel (Source, Range, Flatten, Outlier, Best-5)
- Tab view with 6 analysis tabs
- Export functionality
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QFont, QColor, QIcon, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QProgressBar, QStatusBar,
    QRadioButton, QButtonGroup, QFrame,
)

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from ..core.data_loader import load_recipe, load_dataset, DataSet, RecipeData, POSITION_LABELS
from ..core.analyzer import analyze_recipe, AnalysisResult, get_summary_table
from ..core.flatten import FlattenProcessor
from ..visualization.plot_manager import (
    create_profile_overlay_figure,
    create_flatten_preview_figure,
    create_saturation_trend_figure,
    create_wafer_map_figure,
    create_best5_comparison_figure,
)
from ..visualization.report_generator import (
    export_summary_csv, export_avg_line_csv, export_all_lines_csv, export_checklist,
)

# --- Style ---
DARK_STYLE = """
QMainWindow, QWidget { background-color: #1e1e2e; color: #cdd6f4; }
QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 8px;
            padding-top: 14px; font-weight: bold; color: #89b4fa; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
              border-radius: 4px; padding: 6px 16px; font-size: 12px; }
QPushButton:hover { background-color: #45475a; border: 1px solid #89b4fa; }
QPushButton:pressed { background-color: #585b70; }
QPushButton#export_btn { background-color: #1e66f5; color: white; font-weight: bold; }
QPushButton#export_btn:hover { background-color: #2e7fff; }
QPushButton#load_btn { background-color: #40a02b; color: white; font-weight: bold; }
QPushButton#load_btn:hover { background-color: #50c03b; }
QComboBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
            border-radius: 4px; padding: 4px 8px; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background-color: #313244; color: #cdd6f4;
                               selection-background-color: #45475a; }
QSpinBox, QDoubleSpinBox { background-color: #313244; color: #cdd6f4;
                            border: 1px solid #45475a; border-radius: 4px; padding: 3px; }
QTabWidget::pane { border: 1px solid #45475a; background-color: #1e1e2e; }
QTabBar::tab { background-color: #313244; color: #a6adc8; padding: 8px 16px;
               border: 1px solid #45475a; border-bottom: none; border-radius: 4px 4px 0 0; }
QTabBar::tab:selected { background-color: #1e1e2e; color: #89b4fa; border-bottom: 2px solid #89b4fa; }
QTabBar::tab:hover { background-color: #45475a; }
QTableWidget { background-color: #181825; color: #cdd6f4; gridline-color: #313244;
               border: 1px solid #45475a; font-size: 11px; }
QTableWidget::item { padding: 3px; }
QTableWidget::item:selected { background-color: #45475a; }
QHeaderView::section { background-color: #313244; color: #89b4fa; padding: 4px;
                        border: 1px solid #45475a; font-weight: bold; font-size: 11px; }
QTreeWidget { background-color: #181825; color: #cdd6f4; border: 1px solid #45475a; }
QTreeWidget::item:hover { background-color: #313244; }
QTreeWidget::item:selected { background-color: #45475a; }
QProgressBar { background-color: #313244; border: 1px solid #45475a; border-radius: 4px;
               text-align: center; color: #cdd6f4; }
QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
QStatusBar { background-color: #181825; color: #a6adc8; border-top: 1px solid #313244; }
QLabel#spec_pass { color: #a6e3a1; font-weight: bold; font-size: 14px; }
QLabel#spec_fail { color: #f38ba8; font-weight: bold; font-size: 14px; }
QScrollBar:vertical { background: #181825; width: 10px; }
QScrollBar::handle:vertical { background: #45475a; border-radius: 5px; min-height: 20px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QCheckBox { color: #cdd6f4; }
QRadioButton { color: #cdd6f4; }
"""


class LoadWorker(QThread):
    """Background worker for loading data."""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, path: str, signal_source: str = "Height"):
        super().__init__()
        self.path = path
        self.signal_source = signal_source

    def run(self):
        try:
            self.progress.emit("Loading data...")
            recipe = load_recipe(self.path, signal_source=self.signal_source)
            self.progress.emit(f"Loaded {recipe.repeat_count} repeats.")
            self.finished.emit(recipe)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sliding Stage OPM Repeatability Analyzer")
        self.setMinimumSize(1200, 800)
        self.resize(1440, 900)

        # State
        self.recipe: Optional[RecipeData] = None
        self.result: Optional[AnalysisResult] = None
        self.flatten_proc = FlattenProcessor()
        self._worker: Optional[LoadWorker] = None

        self._setup_ui()
        self.setStyleSheet(DARK_STYLE)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 0)

        # --- Top: Data Loading ---
        load_group = self._create_load_panel()
        layout.addWidget(load_group)

        # --- Main: Splitter (Settings | Tabs) ---
        splitter = QSplitter(Qt.Horizontal)

        # Left: Settings
        settings_widget = self._create_settings_panel()
        splitter.addWidget(settings_widget)

        # Right: Tabs
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(600)

        # Tab 1: Profile Charts
        self.profile_canvas = FigureCanvas(Figure(figsize=(12, 9)))
        self.tabs.addTab(self.profile_canvas, "📊 Profile Charts")

        # Tab 2: Summary Table
        self.summary_table = self._create_summary_table()
        self.tabs.addTab(self.summary_table, "📋 Summary Table")

        # Tab 3: Flatten
        self.flatten_widget = self._create_flatten_tab()
        self.tabs.addTab(self.flatten_widget, "🔧 Flatten")

        # Tab 4: Saturation Trend
        self.trend_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.tabs.addTab(self.trend_canvas, "📈 Saturation Trend")

        # Tab 5: Wafer Map
        self.wafer_canvas = FigureCanvas(Figure(figsize=(8, 7)))
        self.tabs.addTab(self.wafer_canvas, "🗺️ Wafer Map")

        # Tab 6: Best-5
        self.best5_canvas = FigureCanvas(Figure(figsize=(12, 6)))
        self.tabs.addTab(self.best5_canvas, "⭐ Best-5 Window")

        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 1000])

        layout.addWidget(splitter)

        # --- Status Bar ---
        self.statusBar().showMessage("Ready. Select a data folder to begin.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _create_load_panel(self) -> QGroupBox:
        group = QGroupBox("Data Loading")
        layout = QHBoxLayout(group)

        self.path_label = QLabel("No data loaded")
        self.path_label.setStyleSheet("color: #a6adc8; font-size: 12px;")
        layout.addWidget(self.path_label, 1)

        self.load_btn = QPushButton("📂 Open Folder")
        self.load_btn.setObjectName("load_btn")
        self.load_btn.setFixedHeight(32)
        self.load_btn.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.load_btn)

        return group

    def _create_settings_panel(self) -> QWidget:
        widget = QWidget()
        widget.setFixedWidth(250)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        # Signal Source
        source_group = QGroupBox("Signal Source")
        source_layout = QVBoxLayout(source_group)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Height", "Z Drive"])
        source_layout.addWidget(self.source_combo)
        layout.addWidget(source_group)

        # Range
        range_group = QGroupBox("Range")
        range_layout = QVBoxLayout(range_group)
        self.range_combo = QComboBox()
        self.range_combo.addItems(["25mm", "10mm", "5mm", "1mm"])
        range_layout.addWidget(self.range_combo)
        layout.addWidget(range_group)

        # Best-5 Window
        best5_group = QGroupBox("Best-5 Window")
        best5_layout = QVBoxLayout(best5_group)
        h = QHBoxLayout()
        h.addWidget(QLabel("Window Size:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(2, 20)
        self.window_spin.setValue(5)
        h.addWidget(self.window_spin)
        best5_layout.addLayout(h)
        layout.addWidget(best5_group)

        # Spec Info
        spec_group = QGroupBox("Spec Judgment")
        spec_layout = QVBoxLayout(spec_group)
        self.spec_label = QLabel("—")
        self.spec_label.setStyleSheet("font-size: 13px;")
        self.spec_label.setAlignment(Qt.AlignCenter)
        spec_layout.addWidget(self.spec_label)
        self.spec_detail = QLabel("")
        self.spec_detail.setStyleSheet("font-size: 10px; color: #a6adc8;")
        self.spec_detail.setWordWrap(True)
        spec_layout.addWidget(self.spec_detail)
        layout.addWidget(spec_group)

        # Data Info
        info_group = QGroupBox("Data Info")
        info_layout = QVBoxLayout(info_group)
        self.info_tree = QTreeWidget()
        self.info_tree.setHeaderLabels(["Property", "Value"])
        self.info_tree.setMaximumHeight(200)
        self.info_tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        info_layout.addWidget(self.info_tree)
        layout.addWidget(info_group)

        # Actions
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)

        self.analyze_btn = QPushButton("▶ Analyze")
        self.analyze_btn.setStyleSheet("background-color: #1e66f5; color: white; font-weight: bold;")
        self.analyze_btn.clicked.connect(self._on_analyze)
        self.analyze_btn.setEnabled(False)
        action_layout.addWidget(self.analyze_btn)

        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.setObjectName("export_btn")
        self.export_btn.clicked.connect(self._on_export)
        self.export_btn.setEnabled(False)
        action_layout.addWidget(self.export_btn)

        layout.addWidget(action_group)
        layout.addStretch()

        return widget

    def _create_summary_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "Range", "Position", "Rep. Max (nm)", "Rep. 1σ (nm)",
            "OPM Max (nm)", "OPM 1σ (nm)"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setAlternatingRowColors(True)
        return table

    def _create_flatten_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Controls
        ctrl_layout = QHBoxLayout()

        ctrl_layout.addWidget(QLabel("Position:"))
        self.flat_pos_combo = QComboBox()
        self.flat_pos_combo.addItems(POSITION_LABELS)
        ctrl_layout.addWidget(self.flat_pos_combo)

        ctrl_layout.addWidget(QLabel("Repeat:"))
        self.flat_rep_combo = QComboBox()
        ctrl_layout.addWidget(self.flat_rep_combo)

        ctrl_layout.addWidget(QLabel("Order:"))
        self.flat_order_combo = QComboBox()
        self.flat_order_combo.addItems([str(i) for i in range(13)])
        self.flat_order_combo.setCurrentIndex(1)
        ctrl_layout.addWidget(self.flat_order_combo)

        ctrl_layout.addWidget(QLabel("Edge %:"))
        self.flat_edge_spin = QDoubleSpinBox()
        self.flat_edge_spin.setRange(0, 10)
        self.flat_edge_spin.setValue(1.0)
        self.flat_edge_spin.setSingleStep(0.5)
        ctrl_layout.addWidget(self.flat_edge_spin)

        self.flat_execute_btn = QPushButton("Execute")
        self.flat_execute_btn.setStyleSheet("background-color: #40a02b; color: white;")
        self.flat_execute_btn.clicked.connect(self._on_flatten_execute)
        ctrl_layout.addWidget(self.flat_execute_btn)

        self.flat_undo_btn = QPushButton("Undo")
        self.flat_undo_btn.clicked.connect(self._on_flatten_undo)
        self.flat_undo_btn.setEnabled(False)
        ctrl_layout.addWidget(self.flat_undo_btn)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Stats
        self.flat_stats_label = QLabel("")
        self.flat_stats_label.setStyleSheet("font-size: 11px; color: #a6adc8;")
        layout.addWidget(self.flat_stats_label)

        # Canvas
        self.flatten_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout.addWidget(self.flatten_canvas)

        return widget

    # ─── Slots ───────────────────────────────────────────────

    def _on_load_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Recipe Data Folder",
            str(Path("data").resolve()) if Path("data").is_dir() else ""
        )
        if not folder:
            return

        signal = self.source_combo.currentText()
        self.load_btn.setEnabled(False)
        self.statusBar().showMessage(f"Loading {folder}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate

        self._worker = LoadWorker(folder, signal)
        self._worker.finished.connect(self._on_load_finished)
        self._worker.error.connect(self._on_load_error)
        self._worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self._worker.start()

    def _on_load_finished(self, recipe: RecipeData):
        self.recipe = recipe
        self.load_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.path_label.setText(f"📂 {recipe.directory} — {recipe.range_label} "
                                f"({recipe.repeat_count} repeats)")

        # Update info tree
        self.info_tree.clear()
        info = QTreeWidgetItem(["Range", recipe.range_label])
        self.info_tree.addTopLevelItem(info)
        info2 = QTreeWidgetItem(["Repeats", str(recipe.repeat_count)])
        self.info_tree.addTopLevelItem(info2)
        for r in recipe.repeats:
            ritem = QTreeWidgetItem([f"Repeat {r.repeat_no}", r.directory.name])
            ritem.addChild(QTreeWidgetItem(["Profiles", str(len(r.profiles))]))
            ritem.addChild(QTreeWidgetItem(["Lot ID", r.lot_id or "—"]))
            self.info_tree.addTopLevelItem(ritem)

        # Update flatten repeat combo
        self.flat_rep_combo.clear()
        self.flat_rep_combo.addItems([f"Repeat {r.repeat_no}" for r in recipe.repeats])

        # Update range combo
        idx = self.range_combo.findText(recipe.range_label)
        if idx >= 0:
            self.range_combo.setCurrentIndex(idx)

        self.statusBar().showMessage(
            f"Loaded: {recipe.range_label}, {recipe.repeat_count} repeats, "
            f"{sum(len(r.profiles) for r in recipe.repeats)} profiles")

        # Auto-analyze
        self._on_analyze()

    def _on_load_error(self, msg: str):
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Load Error", msg)
        self.statusBar().showMessage("Load failed.")

    def _on_analyze(self):
        if not self.recipe:
            return

        window_size = self.window_spin.value()
        self.result = analyze_recipe(self.recipe, window_size=window_size)

        # Update summary table
        self._update_summary_table()

        # Update spec judgment
        self._update_spec_display()

        # Update all chart tabs
        self._update_profile_chart()
        self._update_trend_chart()
        self._update_wafer_map()
        self._update_best5_chart()

        self.export_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"Analysis complete. Best window: R{self.result.best_window.repeat_range if self.result.best_window else '?'}")

    def _update_summary_table(self):
        if not self.result:
            return

        rows = get_summary_table(self.result, use_best_window=True)
        self.summary_table.setRowCount(len(rows))

        for i, row in enumerate(rows):
            for j, key in enumerate(["Range", "Position", "Rep. Max (nm)",
                                      "Rep. 1σ (nm)", "OPM Max (nm)", "OPM 1σ (nm)"]):
                val = row.get(key, "")
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)

                # Color coding for total rows
                if row.get("Range") == "Total":
                    item.setBackground(QColor("#313244"))

                self.summary_table.setItem(i, j, item)

    def _update_spec_display(self):
        if not self.result:
            return

        if self.result.spec_pass is not None:
            if self.result.spec_pass:
                self.spec_label.setText("✅ PASS")
                self.spec_label.setObjectName("spec_pass")
            else:
                self.spec_label.setText("❌ FAIL")
                self.spec_label.setObjectName("spec_fail")
            self.spec_label.setStyle(self.spec_label.style())

            bw = self.result.best_window
            detail = f"Spec: {self.result.spec_limit} nm\n"
            if bw:
                detail += f"Window: R{bw.repeat_range}\n"
                detail += f"Mean Rep.1σ: {bw.mean_rep_1sigma:.3f} nm"
            self.spec_detail.setText(detail)
        else:
            self.spec_label.setText("—")
            self.spec_detail.setText("")

    @staticmethod
    def _update_canvas(canvas: FigureCanvas, new_fig: Figure):
        """Safely replace the figure on a canvas, preventing rendering ghosts."""
        old_fig = canvas.figure
        # Detach old figure and close it to free resources
        if old_fig is not new_fig:
            plt.close(old_fig)

        # Attach new figure to this canvas
        new_fig.set_canvas(canvas)
        canvas.figure = new_fig

        # Match DPI so the pixel buffer is the right size
        new_fig.set_dpi(canvas.figure.get_dpi())

        # Force the figure to resize to the current canvas geometry
        w = canvas.width()
        h = canvas.height()
        if w > 0 and h > 0:
            new_fig.set_size_inches(w / new_fig.get_dpi(), h / new_fig.get_dpi())

        canvas.draw_idle()
        canvas.update()

    def _update_profile_chart(self):
        if not self.recipe:
            return
        fig = create_profile_overlay_figure(self.recipe, figsize=(12, 9))
        self._update_canvas(self.profile_canvas, fig)

    def _update_trend_chart(self):
        if not self.result:
            return
        fig = create_saturation_trend_figure(self.result, figsize=(10, 6))
        self._update_canvas(self.trend_canvas, fig)

    def _update_wafer_map(self):
        if not self.result:
            return
        fig = create_wafer_map_figure(self.result, metric="rep_max", figsize=(8, 7))
        self._update_canvas(self.wafer_canvas, fig)

    def _update_best5_chart(self):
        if not self.result:
            return
        fig = create_best5_comparison_figure(self.result, figsize=(12, 6))
        self._update_canvas(self.best5_canvas, fig)

    def _on_flatten_execute(self):
        if not self.recipe:
            QMessageBox.warning(self, "Warning", "Load data first.")
            return

        pos = self.flat_pos_combo.currentText()
        rep_idx = self.flat_rep_combo.currentIndex()
        order = int(self.flat_order_combo.currentText())
        edge_pct = self.flat_edge_spin.value()

        if rep_idx < 0 or rep_idx >= len(self.recipe.repeats):
            return

        repeat = self.recipe.repeats[rep_idx]
        if pos not in repeat.profiles:
            QMessageBox.warning(self, "Warning", f"No profile for {pos} in Repeat {rep_idx+1}")
            return

        profile = repeat.profiles[pos]
        result = self.flatten_proc.flatten(
            profile.z_nm, profile.x_mm, order=order, edge_percent=edge_pct
        )

        # Update stats
        self.flat_stats_label.setText(
            f"Order {order} | OPM: {result.opm_before:.3f} → {result.opm_after:.3f} nm | "
            f"RMS: {result.rms_before:.3f} → {result.rms_after:.3f} nm | "
            f"Edge: {edge_pct}%"
        )

        # Plot
        fig = create_flatten_preview_figure(result, profile.x_mm, figsize=(10, 8))
        self._update_canvas(self.flatten_canvas, fig)

        self.flat_undo_btn.setEnabled(self.flatten_proc.can_undo)

    def _on_flatten_undo(self):
        prev = self.flatten_proc.undo()
        if prev and self.recipe:
            pos = self.flat_pos_combo.currentText()
            rep_idx = self.flat_rep_combo.currentIndex()
            if rep_idx >= 0 and rep_idx < len(self.recipe.repeats):
                profile = self.recipe.repeats[rep_idx].profiles.get(pos)
                if profile:
                    fig = create_flatten_preview_figure(prev, profile.x_mm)
                    self._update_canvas(self.flatten_canvas, fig)
                    self.flat_stats_label.setText(
                        f"Undo → Order {prev.order} | OPM: {prev.opm_after:.3f} nm")
        self.flat_undo_btn.setEnabled(self.flatten_proc.can_undo)

    def _on_export(self):
        if not self.result or not self.recipe:
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return

        base = Path(folder)
        try:
            export_summary_csv(self.result, base / "summary.csv")
            export_avg_line_csv(self.recipe, base / "avg_lines.csv")
            export_checklist(self.result, base / "checklist.txt")

            # Save charts
            for name, canvas in [("profiles", self.profile_canvas),
                                  ("trend", self.trend_canvas),
                                  ("wafer_map", self.wafer_canvas),
                                  ("best5", self.best5_canvas)]:
                canvas.figure.savefig(str(base / f"{name}.png"), dpi=150,
                                      facecolor="#1e1e2e", bbox_inches="tight")

            QMessageBox.information(self, "Export", f"Results exported to:\n{folder}")
            self.statusBar().showMessage(f"Exported to {folder}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


def run_app():
    """Launch the application."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()

    if not QApplication.instance():
        sys.exit(app.exec())
    else:
        app.exec()
