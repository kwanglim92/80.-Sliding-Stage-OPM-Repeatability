"""PySide6 GUI for Sliding Stage OPM Repeatability Analyzer.

UX Flow:
    1. Click "Open Folder" → select root data folder or single recipe folder
    2. Auto-detect Range → auto-analyze → display results
    3. Switch between recipes via Range selector
    4. Change Signal Source → auto-reload
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QProgressBar,
    QFrame, QGridLayout, QScrollArea,
)

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from ..core.data_loader import (
    load_recipe, load_dataset, DataSet, RecipeData, POSITION_LABELS,
    POSITION_GRID, _detect_range_mm,
)
from ..core.analyzer import analyze_recipe, AnalysisResult, get_summary_table
from ..core.flatten import FlattenProcessor
from ..core.time_analysis import extract_recipe_timing, RecipeTiming, format_timing_summary
from ..core.ball_screw_analyzer import (
    analyze_ball_screw, BallScrewAnalysisResult, get_dishing_matrix,
    SPEC_DISHING, POSITION_LABELS as BS_POSITION_LABELS,
)
from ..visualization.plot_manager import (
    create_profile_overlay_figure,
    create_flatten_preview_figure,
    create_saturation_trend_figure,
    create_wafer_map_figure,
    create_best5_comparison_figure,
)
from ..visualization.report_generator import (
    export_summary_csv, export_avg_line_csv, export_all_lines_csv, export_checklist,
    export_ball_screw_csv,
)

# --- Style ---
DARK_STYLE = """
QMainWindow, QWidget { background-color: #1e1e2e; color: #cdd6f4; }
QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 8px;
            padding-top: 14px; font-weight: bold; color: #89b4fa; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
              border-radius: 4px; padding: 6px 16px; font-size: 12px;
              min-width: 60px; min-height: 24px; }
QPushButton:hover { background-color: #45475a; border: 1px solid #89b4fa; }
QPushButton:pressed { background-color: #585b70; }
QPushButton#export_btn { background-color: #1e66f5; color: white; font-weight: bold; }
QPushButton#export_btn:hover { background-color: #2e7fff; }
QPushButton#load_btn { background-color: #40a02b; color: white; font-weight: bold; }
QPushButton#load_btn:hover { background-color: #50c03b; }
QComboBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
            border-radius: 4px; padding: 4px 8px; min-height: 22px; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background-color: #313244; color: #cdd6f4;
                               selection-background-color: #45475a; }
QSpinBox, QDoubleSpinBox { background-color: #313244; color: #cdd6f4;
                            border: 1px solid #45475a; border-radius: 4px;
                            padding: 3px; min-height: 22px; min-width: 60px; }
QTabWidget::pane { border: 1px solid #45475a; background-color: #1e1e2e; }
QTabBar::tab { background-color: #313244; color: #a6adc8; padding: 8px 16px;
               border: 1px solid #45475a; border-bottom: none; border-radius: 4px 4px 0 0;
               font-size: 12px; }
QTabBar::tab:selected { background-color: #1e1e2e; color: #89b4fa;
                         border-bottom: 2px solid #89b4fa; }
QTabBar::tab:hover { background-color: #45475a; }
QTableWidget { background-color: #181825; color: #cdd6f4; gridline-color: #313244;
               border: 1px solid #45475a; }
QTableWidget::item { padding: 4px; }
QTableWidget::item:selected { background-color: #45475a; }
QHeaderView::section { background-color: #313244; color: #89b4fa; padding: 6px;
                        border: 1px solid #45475a; font-weight: bold; }
QTreeWidget { background-color: #181825; color: #cdd6f4; border: 1px solid #45475a; }
QTreeWidget::item:hover { background-color: #313244; }
QTreeWidget::item:selected { background-color: #45475a; }
QProgressBar { background-color: #313244; border: 1px solid #45475a; border-radius: 4px;
               text-align: center; color: #cdd6f4; }
QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
QStatusBar { background-color: #181825; color: #a6adc8; border-top: 1px solid #313244; }
QScrollBar:vertical { background: #181825; width: 10px; }
QScrollBar::handle:vertical { background: #45475a; border-radius: 5px; min-height: 20px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""


class LoadWorker(QThread):
    """Background worker for loading data."""
    finished_single = Signal(object)
    finished_multi = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, path: str, signal_source: str = "Height",
                 multi: bool = False):
        super().__init__()
        self.path = path
        self.signal_source = signal_source
        self.multi = multi

    def run(self):
        try:
            if self.multi:
                self.progress.emit("Loading all recipes...")
                dataset = load_dataset(self.path, signal_source=self.signal_source)
                self.progress.emit(f"Loaded {len(dataset.recipes)} recipes.")
                self.finished_multi.emit(dataset)
            else:
                self.progress.emit("Loading recipe...")
                recipe = load_recipe(self.path, signal_source=self.signal_source)
                self.progress.emit(f"Loaded {recipe.repeat_count} repeats.")
                self.finished_single.emit(recipe)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


def _detect_folder_type(path: Path) -> str:
    """Detect folder type: 'root', 'recipe', or 'unknown'."""
    if _detect_range_mm(path.name) is not None:
        return "recipe"
    for child in path.iterdir():
        if child.is_dir() and _detect_range_mm(child.name) is not None:
            return "root"
    return "unknown"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sliding Stage OPM Repeatability Analyzer")
        self.setMinimumSize(1200, 800)
        self.resize(1440, 900)

        # State
        self.dataset: Optional[DataSet] = None
        self.current_recipe: Optional[RecipeData] = None
        self.current_result: Optional[AnalysisResult] = None
        self.current_timing: Optional[RecipeTiming] = None
        self.current_bs_result: Optional[BallScrewAnalysisResult] = None
        self.flatten_proc = FlattenProcessor()
        self._worker: Optional[LoadWorker] = None
        self._loaded_path: Optional[str] = None
        self._block_range_signal = False

        self._setup_ui()
        self.setStyleSheet(DARK_STYLE)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 0)

        # Top: Data Loading
        load_group = self._create_load_panel()
        layout.addWidget(load_group)

        # Main: Splitter (Settings | Tabs)
        splitter = QSplitter(Qt.Horizontal)

        # Left: Settings
        settings_widget = self._create_settings_panel()
        splitter.addWidget(settings_widget)

        # Right: Tabs (NO EMOJIS)
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(600)

        self.profile_canvas = FigureCanvas(Figure(figsize=(12, 9)))
        self.tabs.addTab(self.profile_canvas, "Profile Charts")

        self.summary_table = self._create_summary_table()
        self.tabs.addTab(self.summary_table, "Summary Table")

        self.flatten_widget = self._create_flatten_tab()
        self.tabs.addTab(self.flatten_widget, "Flatten")

        self.trend_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.tabs.addTab(self.trend_canvas, "Saturation Trend")

        self.wafer_canvas = FigureCanvas(Figure(figsize=(8, 7)))
        self.tabs.addTab(self.wafer_canvas, "Wafer Map")

        self.best5_canvas = FigureCanvas(Figure(figsize=(12, 6)))
        self.tabs.addTab(self.best5_canvas, "Best-5 Window")

        self.time_widget = self._create_time_tab()
        self.tabs.addTab(self.time_widget, "Time Analysis")

        # Ball Screw Pitch tab
        self.bs_widget = self._create_ball_screw_tab()
        self.tabs.addTab(self.bs_widget, "Ball Screw Pitch")

        # Remark tab (Export + notes)
        self.remark_widget = self._create_remark_tab()
        self.tabs.addTab(self.remark_widget, "Remark")

        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 1000])

        layout.addWidget(splitter)

        # Status Bar
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

        self.load_btn = QPushButton("Open Folder")
        self.load_btn.setObjectName("load_btn")
        self.load_btn.setFixedHeight(32)
        self.load_btn.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.load_btn)

        return group

    def _create_settings_panel(self) -> QWidget:
        widget = QWidget()
        widget.setFixedWidth(260)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Signal Source
        source_group = QGroupBox("Signal Source")
        source_layout = QVBoxLayout(source_group)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Height", "Z Drive"])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.source_combo)
        layout.addWidget(source_group)

        # Range Selector
        range_group = QGroupBox("Recipe Range")
        range_layout = QVBoxLayout(range_group)
        self.range_combo = QComboBox()
        self.range_combo.setStyleSheet(
            "QComboBox { font-size: 13px; font-weight: bold; padding: 6px; }")
        self.range_combo.currentTextChanged.connect(self._on_range_changed)
        range_layout.addWidget(self.range_combo)
        self.range_info_label = QLabel("")
        self.range_info_label.setStyleSheet("font-size: 10px; color: #a6adc8;")
        range_layout.addWidget(self.range_info_label)
        layout.addWidget(range_group)

        # Best-5 Window — use QFrame instead of QGroupBox to avoid clipping
        best5_frame = QFrame()
        best5_frame.setStyleSheet(
            "QFrame#best5Frame { border: 1px solid #45475a; border-radius: 6px; }")
        best5_frame.setObjectName("best5Frame")
        best5_inner = QVBoxLayout(best5_frame)
        best5_inner.setContentsMargins(10, 6, 10, 8)
        best5_inner.setSpacing(6)
        best5_title = QLabel("Best-5 Window")
        best5_title.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 12px;")
        best5_inner.addWidget(best5_title)
        best5_row = QHBoxLayout()
        best5_row.setSpacing(8)
        ws_label = QLabel("Window Size:")
        ws_label.setStyleSheet("font-size: 12px;")
        best5_row.addWidget(ws_label)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(2, 20)
        self.window_spin.setValue(5)
        self.window_spin.setFixedSize(80, 32)
        self.window_spin.setStyleSheet(
            "QSpinBox { padding: 4px 6px; font-size: 14px; }"
            "QSpinBox::up-button { width: 20px; }"
            "QSpinBox::down-button { width: 20px; }")
        self.window_spin.valueChanged.connect(self._on_reanalyze)
        best5_row.addWidget(self.window_spin)
        best5_row.addStretch()
        best5_inner.addLayout(best5_row)
        layout.addWidget(best5_frame)

        # Spec Judgment — redesigned with equipment type + dual spec
        spec_frame = QFrame()
        spec_frame.setStyleSheet(
            "QFrame#specFrame { border: 1px solid #45475a; border-radius: 6px; }")
        spec_frame.setObjectName("specFrame")
        spec_inner = QVBoxLayout(spec_frame)
        spec_inner.setContentsMargins(10, 6, 10, 8)
        spec_inner.setSpacing(4)

        spec_title_row = QHBoxLayout()
        spec_title = QLabel("Spec Judgment")
        spec_title.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 12px;")
        spec_title_row.addWidget(spec_title)
        spec_title_row.addStretch()

        # Help button — OS standard info icon
        from PySide6.QtWidgets import QStyle, QToolTip
        from PySide6.QtCore import QSize
        _spec_tooltip = (
            "<b>OPM Repeatability</b>: Based on Rep. 1\u03c3<br>"
            "<b>Max OPM</b>: Based on maximum OPM value<br><br>"
            "Both must PASS to qualify.<br>"
            "Click for full spec reference table.")
        self.spec_help_btn = QPushButton()
        self.spec_help_btn.setIcon(self.style().standardIcon(
            QStyle.StandardPixmap.SP_MessageBoxInformation))
        self.spec_help_btn.setIconSize(QSize(16, 16))
        self.spec_help_btn.setFixedSize(22, 22)
        self.spec_help_btn.setCursor(Qt.WhatsThisCursor)
        self.spec_help_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; }"
            "QPushButton:hover { background: #181825; border-radius: 11px; }")
        self.spec_help_btn.setToolTip(_spec_tooltip)
        self.spec_help_btn.clicked.connect(
            lambda checked=False, b=self.spec_help_btn, t=_spec_tooltip:
                QToolTip.showText(b.mapToGlobal(
                    b.rect().bottomLeft()), t, b, b.rect(), 10000))
        self.spec_help_btn.clicked.connect(self._show_spec_info_popup)
        spec_title_row.addWidget(self.spec_help_btn)
        spec_inner.addLayout(spec_title_row)

        # Equipment type radio buttons
        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        radio_style = (
            "QRadioButton { font-size: 12px; font-weight: bold; color: #cdd6f4;"
            "spacing: 6px; padding: 2px 4px; }"
            "QRadioButton::indicator { width: 14px; height: 14px; }"
            "QRadioButton::indicator:checked { "
            "background-color: #89b4fa; border: 2px solid #b4befe; border-radius: 8px; }"
            "QRadioButton::indicator:unchecked { "
            "background-color: #313244; border: 2px solid #585b70; border-radius: 8px; }")
        equip_row = QHBoxLayout()
        equip_row.setSpacing(6)
        self.equip_group = QButtonGroup(self)
        self.radio_iso = QRadioButton("Isolated AE")
        self.radio_dw = QRadioButton("Double Walled AE")
        self.radio_iso.setStyleSheet(radio_style)
        self.radio_dw.setStyleSheet(radio_style)
        self.radio_iso.setChecked(True)  # Default: Isolated AE
        self.equip_group.addButton(self.radio_iso)
        self.equip_group.addButton(self.radio_dw)
        self.radio_iso.toggled.connect(self._on_equipment_changed)
        equip_row.addWidget(self.radio_iso)
        equip_row.addWidget(self.radio_dw)
        equip_row.addStretch()
        spec_inner.addLayout(equip_row)

        # Spec value lines (vertical, left-aligned)
        self.spec_lines_label = QLabel("\u2014")
        self.spec_lines_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.spec_lines_label.setStyleSheet(
            "font-size: 12px; color: #cdd6f4; padding: 4px 2px;")
        self.spec_lines_label.setWordWrap(True)
        spec_inner.addWidget(self.spec_lines_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #45475a;")
        spec_inner.addWidget(sep)

        # Overall verdict
        self.spec_verdict_label = QLabel("\u2014")
        self.spec_verdict_label.setAlignment(Qt.AlignCenter)
        self.spec_verdict_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 4px;")
        spec_inner.addWidget(self.spec_verdict_label)

        layout.addWidget(spec_frame)

        # Data Info — stretch to fill remaining space
        info_group = QGroupBox("Data Info")
        info_layout = QVBoxLayout(info_group)
        self.info_tree = QTreeWidget()
        self.info_tree.setHeaderLabels(["Property", "Value"])
        self.info_tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        info_layout.addWidget(self.info_tree)
        layout.addWidget(info_group, 1)  # stretch factor = 1 → fills remaining space

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
        # Bigger font for readability
        table.setStyleSheet("""
            QTableWidget { font-size: 13px; }
            QTableWidget::item { padding: 6px; }
            QHeaderView::section { font-size: 13px; padding: 8px; }
        """)
        table.verticalHeader().setDefaultSectionSize(32)
        return table

    def _create_flatten_tab(self) -> QWidget:
        """Flatten tab — single row controls for maximum chart space."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Controls — single compact row
        ctrl_row = QHBoxLayout()
        ctrl_row.setContentsMargins(4, 4, 4, 4)
        ctrl_row.setSpacing(6)

        ctrl_row.addWidget(QLabel("Pos:"))
        self.flat_pos_combo = QComboBox()
        self.flat_pos_combo.addItems(POSITION_LABELS)
        self.flat_pos_combo.setFixedWidth(70)
        ctrl_row.addWidget(self.flat_pos_combo)

        ctrl_row.addWidget(QLabel("Rep:"))
        self.flat_rep_combo = QComboBox()
        self.flat_rep_combo.setFixedWidth(90)
        ctrl_row.addWidget(self.flat_rep_combo)

        ctrl_row.addWidget(QLabel("Ord:"))
        self.flat_order_combo = QComboBox()
        self.flat_order_combo.addItems([str(i) for i in range(13)])
        self.flat_order_combo.setCurrentIndex(1)
        self.flat_order_combo.setFixedWidth(50)
        ctrl_row.addWidget(self.flat_order_combo)

        ctrl_row.addWidget(QLabel("Edge%:"))
        self.flat_edge_spin = QDoubleSpinBox()
        self.flat_edge_spin.setRange(0, 10)
        self.flat_edge_spin.setValue(1.0)
        self.flat_edge_spin.setSingleStep(0.5)
        self.flat_edge_spin.setFixedSize(75, 30)
        self.flat_edge_spin.setStyleSheet(
            "QDoubleSpinBox { padding: 3px 4px; font-size: 12px; }"
            "QDoubleSpinBox::up-button { width: 18px; }"
            "QDoubleSpinBox::down-button { width: 18px; }")
        ctrl_row.addWidget(self.flat_edge_spin)

        self.flat_execute_btn = QPushButton("Execute")
        self.flat_execute_btn.setStyleSheet(
            "background-color: #40a02b; color: white; font-weight: bold;"
            "padding: 4px 12px;")
        self.flat_execute_btn.clicked.connect(self._on_flatten_execute)
        ctrl_row.addWidget(self.flat_execute_btn)

        self.flat_undo_btn = QPushButton("Undo")
        self.flat_undo_btn.setStyleSheet("padding: 4px 12px;")
        self.flat_undo_btn.clicked.connect(self._on_flatten_undo)
        self.flat_undo_btn.setEnabled(False)
        ctrl_row.addWidget(self.flat_undo_btn)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Stats (minimal height)
        self.flat_stats_label = QLabel("")
        self.flat_stats_label.setStyleSheet("font-size: 11px; color: #a6adc8; padding: 2px 4px;")
        self.flat_stats_label.setFixedHeight(20)
        layout.addWidget(self.flat_stats_label)

        # Canvas
        self.flatten_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout.addWidget(self.flatten_canvas)

        return widget

    def _create_time_tab(self) -> QWidget:
        """Time Analysis tab — shows per-repeat timing, gaps, estimation."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        # Summary section
        summary_frame = QFrame()
        summary_frame.setStyleSheet(
            "QFrame { background-color: #181825; border-radius: 6px; padding: 12px; }")
        summary_layout = QGridLayout(summary_frame)
        summary_layout.setHorizontalSpacing(24)
        summary_layout.setVerticalSpacing(6)

        self.time_total_label = QLabel("—")
        self.time_total_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa;")
        summary_layout.addWidget(QLabel("Total Duration:"), 0, 0)
        summary_layout.addWidget(self.time_total_label, 0, 1)

        self.time_avg_repeat_label = QLabel("—")
        self.time_avg_repeat_label.setStyleSheet("font-size: 14px; color: #cdd6f4;")
        summary_layout.addWidget(QLabel("Avg per Repeat:"), 1, 0)
        summary_layout.addWidget(self.time_avg_repeat_label, 1, 1)

        self.time_avg_point_label = QLabel("—")
        self.time_avg_point_label.setStyleSheet("font-size: 14px; color: #cdd6f4;")
        summary_layout.addWidget(QLabel("Avg per Point:"), 2, 0)
        summary_layout.addWidget(self.time_avg_point_label, 2, 1)

        self.time_continuous_label = QLabel("—")
        self.time_continuous_label.setStyleSheet("font-size: 14px;")
        summary_layout.addWidget(QLabel("Continuity:"), 3, 0)
        summary_layout.addWidget(self.time_continuous_label, 3, 1)

        # Estimation section
        summary_layout.addWidget(QLabel(""), 4, 0)  # spacer
        est_title = QLabel("Estimate for N Repeats:")
        est_title.setStyleSheet("font-weight: bold; color: #89b4fa;")
        summary_layout.addWidget(est_title, 5, 0, 1, 2)

        est_row = QHBoxLayout()
        self.time_est_spin = QSpinBox()
        self.time_est_spin.setRange(1, 100)
        self.time_est_spin.setValue(10)
        self.time_est_spin.setFixedWidth(80)
        self.time_est_spin.valueChanged.connect(self._update_time_estimate)
        est_row.addWidget(QLabel("Repeat Count:"))
        est_row.addWidget(self.time_est_spin)
        self.time_est_result = QLabel("—")
        self.time_est_result.setStyleSheet("font-size: 16px; font-weight: bold; color: #f9e2af;")
        est_row.addWidget(QLabel("→"))
        est_row.addWidget(self.time_est_result)
        est_row.addStretch()

        est_widget = QWidget()
        est_widget.setLayout(est_row)
        summary_layout.addWidget(est_widget, 6, 0, 1, 2)

        layout.addWidget(summary_frame)

        # Per-repeat table
        self.time_table = QTableWidget()
        self.time_table.setColumnCount(7)
        self.time_table.setHorizontalHeaderLabels([
            "Repeat", "Folder", "Start", "End", "Duration", "Per Point", "Gap"
        ])
        self.time_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.time_table.setStyleSheet("""
            QTableWidget { font-size: 13px; }
            QTableWidget::item { padding: 6px; }
            QHeaderView::section { font-size: 13px; padding: 8px; }
        """)
        self.time_table.verticalHeader().setDefaultSectionSize(30)
        layout.addWidget(self.time_table)

        return widget

    def _create_ball_screw_tab(self) -> QWidget:
        """Ball Screw Pitch analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── Controls row ─────────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        ctrl_row.addWidget(QLabel("Material:"))
        self.bs_material_combo = QComboBox()
        self.bs_material_combo.addItems(["AL (≤6.0 nm)", "SUS (≤4.5 nm)"])
        self.bs_material_combo.setFixedWidth(130)
        ctrl_row.addWidget(self.bs_material_combo)

        from PySide6.QtWidgets import QCheckBox
        self.bs_show_stab_check = QCheckBox("안정화 포인트 표시")
        self.bs_show_stab_check.setChecked(False)
        self.bs_show_stab_check.stateChanged.connect(self._on_bs_filter_changed)
        ctrl_row.addWidget(self.bs_show_stab_check)

        self.bs_analyze_btn = QPushButton("Analyze")
        self.bs_analyze_btn.setStyleSheet(
            "background-color: #40a02b; color: white; font-weight: bold; padding: 4px 14px;")
        self.bs_analyze_btn.clicked.connect(self._on_bs_analyze)
        ctrl_row.addWidget(self.bs_analyze_btn)

        # Verdict badge
        self.bs_verdict_label = QLabel("—")
        self.bs_verdict_label.setAlignment(Qt.AlignCenter)
        self.bs_verdict_label.setFixedWidth(80)
        self.bs_verdict_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; border: 2px solid #45475a;"
            "border-radius: 6px; padding: 4px; color: #a6adc8;")
        ctrl_row.addWidget(self.bs_verdict_label)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # ── Plot area: bar chart (left) + heatmap (right) ────────────────────
        plot_row = QHBoxLayout()
        self.bs_bar_canvas = FigureCanvas(Figure(figsize=(8, 4)))
        self.bs_heatmap_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        plot_row.addWidget(self.bs_bar_canvas, 6)
        plot_row.addWidget(self.bs_heatmap_canvas, 4)
        layout.addLayout(plot_row)

        # ── Summary table ────────────────────────────────────────────────────
        self.bs_table = QTableWidget()
        self.bs_table.setMinimumHeight(180)
        self.bs_table.setMaximumHeight(220)
        self.bs_table.setStyleSheet("""
            QTableWidget { font-size: 12px; }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section { font-size: 12px; padding: 6px;
                background-color: #313244; color: #89b4fa;
                border: 1px solid #45475a; font-weight: bold; }
        """)
        self.bs_table.verticalHeader().setDefaultSectionSize(26)
        layout.addWidget(self.bs_table)

        return widget

    def _create_remark_tab(self) -> QWidget:
        """Remark tab — Export + Usage Guide."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Export section
        export_frame = QFrame()
        export_frame.setStyleSheet(
            "QFrame#exportFrame { border: 1px solid #45475a; border-radius: 6px; }")
        export_frame.setObjectName("exportFrame")
        export_layout = QVBoxLayout(export_frame)
        export_layout.setContentsMargins(16, 12, 16, 12)
        export_layout.setSpacing(8)

        export_title = QLabel("Export Analysis Results")
        export_title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #89b4fa;")
        export_layout.addWidget(export_title)

        export_desc = QLabel(
            "Export summary CSV, average line profiles, spec checklist,\n"
            "and all chart images (PNG) to a selected folder.")
        export_desc.setStyleSheet("font-size: 12px; color: #a6adc8;")
        export_desc.setWordWrap(True)
        export_layout.addWidget(export_desc)

        self.export_btn = QPushButton("Export Results")
        self.export_btn.setObjectName("export_btn")
        self.export_btn.setFixedHeight(40)
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #1e66f5; color: white;"
            "font-weight: bold; font-size: 14px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #2e7fff; }"
            "QPushButton:disabled { background-color: #45475a; color: #6c7086; }")
        self.export_btn.clicked.connect(self._on_export)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)

        layout.addWidget(export_frame)

        # Usage Guide
        from PySide6.QtWidgets import QTextBrowser
        guide = QTextBrowser()
        guide.setOpenExternalLinks(False)
        guide.setStyleSheet(
            "QTextBrowser { background-color: #181825; color: #cdd6f4;"
            "border: 1px solid #45475a; border-radius: 6px;"
            "padding: 12px; font-size: 12px; }")
        guide.setHtml(self._get_usage_guide_html())
        layout.addWidget(guide, 1)

        return widget

    @staticmethod
    def _get_usage_guide_html() -> str:
        return """
        <style>
            h2 { color: #89b4fa; margin-top: 16px; margin-bottom: 4px; font-size: 15px; }
            h3 { color: #f9e2af; margin-top: 12px; margin-bottom: 2px; font-size: 13px; }
            p, li { color: #cdd6f4; font-size: 12px; line-height: 1.5; }
            ul { margin-left: 16px; }
            .metric { color: #a6e3a1; font-weight: bold; }
            .note { color: #fab387; font-style: italic; }
        </style>

        <h2>Sliding Stage OPM Repeatability Analyzer</h2>
        <p>Park Systems Sliding Stage의 OPM 재현성을 분석하는 데스크탑 도구입니다.</p>

        <h3>1. Profile Charts</h3>
        <p>9개 Position (3×3 Grid)의 프로파일 오버레이를 표시합니다.</p>
        <ul>
        <li>각 서브플롯은 해당 Position의 <b>모든 Repeat</b> 프로파일을 겹쳐 보여줍니다.</li>
        <li>겹침이 클수록 재현성이 좋고, 산포가 클수록 편차가 큽니다.</li>
        <li>특정 Repeat에서 이상 프로파일이 보이면 장비 이상 / 환경 변화를 의심합니다.</li>
        </ul>

        <h3>2. Summary Table</h3>
        <p>Position별 통계 테이블입니다. Best-5 Window 기준 데이터를 사용합니다.</p>
        <ul>
        <li><span class='metric'>Rep. Max (nm)</span>: Pixel-wise Range의 최대값 (재현성 지표)</li>
        <li><span class='metric'>Rep. 1σ (nm)</span>: Pixel-wise Range의 표준편차</li>
        <li><span class='metric'>OPM Max (nm)</span>: Profile별 Max-Min 중 최대값</li>
        <li><span class='metric'>OPM 1σ (nm)</span>: Profile별 Max-Min의 표준편차</li>
        <li>하단 Total 행: Mean / Stdev / Max / RMS 요약</li>
        </ul>

        <h3>3. Flatten</h3>
        <p>개별 프로파일에 대해 Polynomial Flattening을 적용합니다.</p>
        <ul>
        <li><b>Position</b> / <b>Repeat</b> 선택 후 <b>Order</b> 설정 (권장: 1차 또는 2차)</li>
        <li><b>Edge%</b>: 양쪽 가장자리 데이터 제외 비율 (기본 1%)</li>
        <li>Execute 클릭 시 Original / Flattened / Histogram 시각화</li>
        <li>OPM 변화량과 RMS 변화량이 Status Bar에 표시됩니다.</li>
        </ul>

        <h3>4. Saturation Trend</h3>
        <p>Repeat 수 증가에 따른 Rep. 1σ Mean 추이를 보여줍니다.</p>
        <ul>
        <li>그래프가 <b>수렴</b>하면 현재 Repeat 수가 충분합니다.</li>
        <li>아직 하강 추세이면 Repeat를 더 늘려야 합니다.</li>
        <li>초기 값이 매우 높다가 급감하는 경우, 첫 Repeat에 이상이 있을 수 있습니다.</li>
        </ul>

        <h3>5. Wafer Map</h3>
        <p>3×3 Grid로 각 Position의 OPM Max를 Heatmap으로 표시합니다.</p>
        <ul>
        <li><b>빨간색</b>: 높은 값 (편차 큼) → 해당 위치 점검 필요</li>
        <li><b>녹색</b>: 낮은 값 (편차 작음) → 양호</li>
        <li>특정 영역에 빨간색이 몰려있으면 Stage의 기계적 문제를 의심합니다.</li>
        </ul>

        <h3>6. Best-5 Window</h3>
        <p>연속된 5개(기본) Repeat 구간 중 최적 구간을 찾습니다.</p>
        <ul>
        <li>선정 기준: <b>Rep. 1σ Mean이 최소</b>인 구간</li>
        <li>좌측 패널의 Window Size를 변경하면 즉시 재분석됩니다.</li>
        <li>그래프: Best Window vs All Repeats의 Position별 비교</li>
        </ul>

        <h3>7. Time Analysis</h3>
        <p>측정 소요 시간을 분석합니다.</p>
        <ul>
        <li>Repeat별 Start/End/Duration 및 포인트당 소요 시간</li>
        <li>연속 측정 여부 확인 (Gap 2분 이상이면 중단이 있었음)</li>
        <li>하단에서 N-repeat에 필요한 <b>예상 소요 시간</b>을 추정합니다.</li>
        <li class='note'>공수 반영 시 활용: 10-repeat는 약 2배, 20-repeat는 약 4배 소요</li>
        </ul>

        <h3>8. Spec Judgment (좌측 패널)</h3>
        <p>장비 타입에 따라 다른 기준으로 PASS/FAIL을 판단합니다.</p>
        <ul>
        <li><b>분리형 (Isolated AE)</b>: Total RMS / Total Max 기준</li>
        <li><b>일체형 (Double Walled AE)</b>: Center(5_CM) 값 기준</li>
        <li>OPM Repeatability + Max OPM 두 항목 <b>모두 PASS</b>해야 합격</li>
        <li><b>?</b> 버튼을 클릭하면 전체 Spec 테이블을 확인할 수 있습니다.</li>
        </ul>
        """

    # ─── Data Loading ────────────────────────────────────────

    def _on_load_clicked(self):
        start_dir = self._loaded_path or ""
        if not start_dir:
            for candidate in [Path("data"), Path(".")]:
                if candidate.is_dir():
                    start_dir = str(candidate.resolve())
                    break

        folder = QFileDialog.getExistingDirectory(
            self, "Select Data Folder (root or recipe)", start_dir)
        if not folder:
            return
        self._start_load(folder)

    def _start_load(self, folder: str):
        path = Path(folder)
        folder_type = _detect_folder_type(path)

        if folder_type == "unknown":
            parent_type = _detect_folder_type(path.parent)
            if parent_type == "recipe":
                QMessageBox.warning(
                    self, "Wrong folder level",
                    f"Selected a repeat folder:\n{path.name}\n\n"
                    f"Please select the recipe folder:\n{path.parent}\n\n"
                    f"Or select the root data folder to load all recipes."
                )
                return
            else:
                QMessageBox.warning(
                    self, "Unrecognized folder",
                    f"Could not detect recipe data in:\n{folder}\n\n"
                    f"Expected folder names like '25mm', '10mm', '5mm', '1mm'\n"
                    f"or a parent folder containing them."
                )
                return

        self._loaded_path = folder
        signal = self.source_combo.currentText()
        is_multi = (folder_type == "root")

        self.load_btn.setEnabled(False)
        self.statusBar().showMessage(f"Loading {'all recipes' if is_multi else 'recipe'}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self._worker = LoadWorker(folder, signal, multi=is_multi)
        self._worker.finished_single.connect(self._on_single_loaded)
        self._worker.finished_multi.connect(self._on_multi_loaded)
        self._worker.error.connect(self._on_load_error)
        self._worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self._worker.start()

    def _on_single_loaded(self, recipe: RecipeData):
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        from ..core.data_loader import DataSet
        self.dataset = DataSet(
            root_directory=recipe.directory.parent,
            recipes={recipe.range_label: recipe}
        )
        self._populate_range_selector()
        self.path_label.setText(
            f"{recipe.directory} — {recipe.range_label} ({recipe.repeat_count} repeats)")

    def _on_multi_loaded(self, dataset: DataSet):
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if not dataset.recipes:
            QMessageBox.warning(self, "No Data", "No recipe data found.")
            return

        self.dataset = dataset
        self._populate_range_selector()

        total = sum(r.repeat_count for r in dataset.recipes.values())
        ranges = ", ".join(dataset.available_ranges)
        self.path_label.setText(f"{dataset.root_directory} — {ranges} ({total} total repeats)")

    def _on_load_error(self, msg: str):
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Load Error", msg)

    def _populate_range_selector(self):
        self._block_range_signal = True
        self.range_combo.clear()
        if self.dataset:
            for label in self.dataset.available_ranges:
                recipe = self.dataset.recipes[label]
                self.range_combo.addItem(f"{label} ({recipe.repeat_count} repeats)")
        self._block_range_signal = False

        if self.range_combo.count() > 0:
            self.range_combo.setCurrentIndex(0)
            self._switch_to_current_range()

    def _on_range_changed(self, text: str):
        if self._block_range_signal or not text:
            return
        self._switch_to_current_range()

    def _switch_to_current_range(self):
        if not self.dataset:
            return

        combo_text = self.range_combo.currentText()
        range_label = combo_text.split(" (")[0] if " (" in combo_text else combo_text

        if range_label not in self.dataset.recipes:
            return

        self.current_recipe = self.dataset.recipes[range_label]
        self.range_info_label.setText(
            f"{self.current_recipe.repeat_count} repeats, "
            f"{sum(len(r.profiles) for r in self.current_recipe.repeats)} profiles")

        self.flat_rep_combo.clear()
        self.flat_rep_combo.addItems(
            [f"Repeat {r.repeat_no}" for r in self.current_recipe.repeats])

        self._update_info_tree()
        self._run_analysis()

    def _on_source_changed(self, source: str):
        if self._loaded_path and self.dataset:
            reply = QMessageBox.question(
                self, "Reload Data?",
                f"Signal source changed to '{source}'.\nReload all data?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._start_load(self._loaded_path)

    def _on_equipment_changed(self):
        """Re-analyze when equipment type radio is toggled."""
        if self.current_recipe:
            self._run_analysis()

    def _show_spec_info_popup(self):
        """Show spec table popup dialog."""
        from PySide6.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QLabel, QPushButton, QHeaderView
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor
        from ..core.analyzer import SPEC_REPEATABILITY, SPEC_MAX_OPM_DW, SPEC_MAX_OPM_ISO

        dlg = QDialog(self)
        dlg.setWindowTitle("Spec Reference Table")
        dlg.setMinimumSize(600, 400)
        dlg.setStyleSheet(
            "QDialog { background-color: #1e1e2e; }"
            "QLabel { color: #cdd6f4; }"
            "QTableWidget { background-color: #181825; color: #cdd6f4;"
            "gridline-color: #313244; border: 1px solid #45475a; font-size: 13px; }"
            "QTableWidget::item { padding: 6px; }"
            "QHeaderView::section { background-color: #313244; color: #89b4fa;"
            "padding: 8px; border: 1px solid #45475a; font-weight: bold; font-size: 13px; }")

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Sliding Stage Spec Limits")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        layout.addWidget(title)

        equip_type = "iso" if self.radio_iso.isChecked() else "dw"
        equip_label = "Isolated AE" if equip_type == "iso" else "Double Walled AE"
        current_label = QLabel(f"Current: {equip_label}")
        current_label.setStyleSheet("font-size: 12px; color: #a6adc8; margin-bottom: 8px;")
        layout.addWidget(current_label)

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels([
            "Range", "OPM\nRepeatability",
            "Max OPM\n(Double Walled)", "Max OPM\n(Isolated)", "Basis"])
        table.horizontalHeader().setMinimumHeight(44)

        ranges = [25, 10, 5, 1]
        table.setRowCount(len(ranges))
        for i, mm in enumerate(ranges):
            table.setItem(i, 0, QTableWidgetItem(f"{mm}mm"))
            table.setItem(i, 1, QTableWidgetItem(f"{SPEC_REPEATABILITY.get(mm, 'N/A')} nm"))
            table.setItem(i, 2, QTableWidgetItem(f"{SPEC_MAX_OPM_DW.get(mm, 'N/A')} nm"))
            table.setItem(i, 3, QTableWidgetItem(f"{SPEC_MAX_OPM_ISO.get(mm, 'N/A')} nm"))

            if equip_type == "dw":
                basis = "Center (5_CM)"
            else:
                basis = "Total RMS / Max"
            table.setItem(i, 4, QTableWidgetItem(basis))

            for j in range(5):
                item = table.item(i, j)
                if item:
                    item.setTextAlignment(Qt.AlignCenter)

        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        layout.addWidget(table)

        # Note
        note = QLabel(
            "\u2022 OPM Repeatability: Based on Rep. 1\u03c3 (DW=Center, ISO=Total RMS)\n"
            "\u2022 Max OPM: Based on max OPM value (DW=Center, ISO=Total Max)\n"
            "\u2022 Both items must PASS to qualify")
        note.setStyleSheet("font-size: 11px; color: #a6adc8; padding: 8px;")
        layout.addWidget(note)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            "QPushButton { background-color: #45475a; color: #cdd6f4;"
            "padding: 8px 24px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background-color: #585b70; }")
        close_btn.clicked.connect(dlg.close)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

        dlg.exec()

    def _on_reanalyze(self):
        if self.current_recipe:
            self._run_analysis()

    # ─── Analysis ────────────────────────────────────────────

    def _run_analysis(self):
        if not self.current_recipe:
            return

        window_size = self.window_spin.value()
        equipment_type = "iso" if self.radio_iso.isChecked() else "dw"
        self.current_result = analyze_recipe(
            self.current_recipe, window_size=window_size,
            equipment_type=equipment_type)

        # Time analysis
        self.current_timing = extract_recipe_timing(self.current_recipe)

        self._update_summary_table()
        self._update_spec_display()
        self._update_profile_chart()
        self._update_trend_chart()
        self._update_wafer_map()
        self._update_best5_chart()
        self._update_time_tab()
        # Reset Ball Screw result when recipe changes (requires explicit Analyze click)
        self.current_bs_result = None
        self._clear_bs_tab()

        self.export_btn.setEnabled(True)

        bw = self.current_result.best_window
        spec_text = ""
        if self.current_result.spec_pass is not None:
            spec_text = "PASS" if self.current_result.spec_pass else "FAIL"
        if bw:
            self.statusBar().showMessage(
                f"Analysis: {self.current_result.range_label} | "
                f"Best: R{bw.repeat_range} | "
                f"Rep.1σ: {bw.mean_rep_1sigma:.3f}nm | {spec_text}")
        else:
            self.statusBar().showMessage(
                f"Analysis: {self.current_result.range_label} | {spec_text}")

    def _update_info_tree(self):
        self.info_tree.clear()
        if not self.current_recipe:
            return
        r = self.current_recipe
        self.info_tree.addTopLevelItem(QTreeWidgetItem(["Range", r.range_label]))
        self.info_tree.addTopLevelItem(QTreeWidgetItem(["Repeats", str(r.repeat_count)]))
        self.info_tree.addTopLevelItem(QTreeWidgetItem([
            "Source", self.source_combo.currentText()]))
        for rep in r.repeats:
            ritem = QTreeWidgetItem([f"Repeat {rep.repeat_no}", rep.directory.name])
            ritem.addChild(QTreeWidgetItem(["Profiles", str(len(rep.profiles))]))
            if rep.lot_id:
                ritem.addChild(QTreeWidgetItem(["Lot ID", rep.lot_id]))
            self.info_tree.addTopLevelItem(ritem)

    # ─── Chart Updates ───────────────────────────────────────

    @staticmethod
    def _update_canvas(canvas: FigureCanvas, new_fig: Figure):
        """Safely replace figure, preventing rendering ghosts."""
        old_fig = canvas.figure
        if old_fig is not new_fig:
            plt.close(old_fig)
        new_fig.set_canvas(canvas)
        canvas.figure = new_fig
        new_fig.set_dpi(canvas.figure.get_dpi())
        w, h = canvas.width(), canvas.height()
        if w > 0 and h > 0:
            new_fig.set_size_inches(w / new_fig.get_dpi(), h / new_fig.get_dpi())
        canvas.draw_idle()
        canvas.update()

    def _update_summary_table(self):
        if not self.current_result:
            return
        rows = get_summary_table(self.current_result, use_best_window=True)
        self.summary_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, key in enumerate(["Range", "Position", "Rep. Max (nm)",
                                      "Rep. 1σ (nm)", "OPM Max (nm)", "OPM 1σ (nm)"]):
                val = row.get(key, "")
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                if row.get("Range") == "Total":
                    item.setBackground(QColor("#313244"))
                    item.setFont(QFont("Segoe UI", 11, QFont.Bold))
                self.summary_table.setItem(i, j, item)

    def _update_spec_display(self):
        """Update spec judgment with dual-spec values and overall verdict."""
        if not self.current_result:
            return

        r = self.current_result
        equip_label = "Isolated AE" if r.equipment_type == "iso" else "Double Walled AE"
        basis = "Total RMS / Max" if r.equipment_type == "iso" else "Center (5_CM)"

        lines = []
        lines.append(f"<b>Type:</b> {equip_label}")
        lines.append(f"<b>Basis:</b> {basis}")

        if r.best_window:
            lines.append(f"<b>Window:</b> R{r.best_window.repeat_range}")

        # OPM Repeatability
        if r.spec_limit is not None:
            val_str = f"{r.spec_value:.3f}" if r.spec_value is not None else "N/A"
            pass_icon = "\u2705" if r.spec_pass else "\u274c"
            lines.append(f"")
            lines.append(f"{pass_icon} <b>Rep. 1\u03c3:</b> {val_str} / {r.spec_limit} nm")

        # Max OPM
        if r.spec_opm_limit is not None:
            val_str = f"{r.spec_opm_value:.3f}" if r.spec_opm_value is not None else "N/A"
            pass_icon = "\u2705" if r.spec_opm_pass else "\u274c"
            lines.append(f"{pass_icon} <b>OPM Max:</b> {val_str} / {r.spec_opm_limit} nm")

        self.spec_lines_label.setText("<br>".join(lines))

        # Overall verdict
        overall = r.overall_pass
        if overall is not None:
            if overall:
                self.spec_verdict_label.setText("PASS")
                self.spec_verdict_label.setStyleSheet(
                    "font-size: 18px; font-weight: bold; color: #a6e3a1; padding: 4px;")
            else:
                self.spec_verdict_label.setText("FAIL")
                self.spec_verdict_label.setStyleSheet(
                    "font-size: 18px; font-weight: bold; color: #f38ba8; padding: 4px;")
        else:
            self.spec_verdict_label.setText("\u2014")
            self.spec_verdict_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; padding: 4px;")

    def _update_profile_chart(self):
        if not self.current_recipe:
            return
        fig = create_profile_overlay_figure(self.current_recipe, figsize=(12, 9))
        self._update_canvas(self.profile_canvas, fig)

        # Store axes → position mapping for double-click
        self._profile_axes_map = {}
        axes = fig.get_axes()
        for pos in POSITION_LABELS:
            r, c = POSITION_GRID[pos]
            idx = r * 3 + c
            if idx < len(axes):
                self._profile_axes_map[id(axes[idx])] = pos

        # Connect double-click (reconnect each time figure is replaced)
        self.profile_canvas.mpl_connect('button_press_event', self._on_profile_dblclick)

    def _on_profile_dblclick(self, event):
        """Open Position Detail Dialog on double-click."""
        if not event.dblclick or event.inaxes is None:
            return
        if not hasattr(self, '_profile_axes_map'):
            return
        pos = self._profile_axes_map.get(id(event.inaxes))
        if pos and self.current_recipe:
            from .position_detail_dialog import PositionDetailDialog
            dlg = PositionDetailDialog(
                pos, self.current_recipe, self.current_result, parent=self)
            dlg.exec()

    def _update_trend_chart(self):
        if not self.current_result:
            return
        fig = create_saturation_trend_figure(self.current_result, figsize=(10, 6))
        self._update_canvas(self.trend_canvas, fig)

    def _update_wafer_map(self):
        if not self.current_result:
            return
        fig = create_wafer_map_figure(self.current_result, metric="rep_max", figsize=(8, 7))
        self._update_canvas(self.wafer_canvas, fig)

    def _update_best5_chart(self):
        if not self.current_result:
            return
        fig = create_best5_comparison_figure(self.current_result, figsize=(12, 6))
        self._update_canvas(self.best5_canvas, fig)

    # ─── Time Analysis ───────────────────────────────────────

    def _update_time_tab(self):
        """Populate Time Analysis tab with timing data."""
        if not self.current_timing:
            return

        t = self.current_timing

        # Summary labels
        td = t.total_duration
        self.time_total_label.setText(t.total_duration_str if td else "—")

        avg_rep = t.avg_repeat_duration
        if avg_rep:
            from ..core.time_analysis import _fmt_duration
            self.time_avg_repeat_label.setText(_fmt_duration(avg_rep))
        else:
            self.time_avg_repeat_label.setText("—")

        avg_pt = t.avg_per_point_sec
        if avg_pt:
            m, s = divmod(int(avg_pt), 60)
            self.time_avg_point_label.setText(f"{m}m {s:02d}s" if m else f"{s}s")
        else:
            self.time_avg_point_label.setText("—")

        if t.is_continuous:
            self.time_continuous_label.setText("Continuous")
            self.time_continuous_label.setStyleSheet("font-size: 14px; color: #a6e3a1;")
        else:
            self.time_continuous_label.setText("Gaps detected")
            self.time_continuous_label.setStyleSheet("font-size: 14px; color: #f9e2af;")

        # Estimation
        self._update_time_estimate()

        # Table
        rows = format_timing_summary(t)
        self.time_table.setRowCount(len(rows))
        cols = ["Repeat", "Folder", "Start", "End", "Duration", "Per Point", "Gap"]
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                val = row.get(col, "")
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                # Highlight gaps
                if col == "Gap" and val and "⚠" in str(val):
                    item.setForeground(QColor("#f9e2af"))
                self.time_table.setItem(i, j, item)

    def _update_time_estimate(self):
        if not self.current_timing:
            return
        n = self.time_est_spin.value()
        est = self.current_timing.estimate_duration(n)
        self.time_est_result.setText(est)

    # ─── Ball Screw Pitch ─────────────────────────────────────

    def _on_bs_analyze(self):
        """Run Ball Screw Pitch analysis on current recipe."""
        if not self.current_recipe:
            QMessageBox.warning(self, "Warning", "Load data first.")
            return

        material_text = self.bs_material_combo.currentText()
        material = "AL" if material_text.startswith("AL") else "SUS"
        signal_source = self.source_combo.currentText()

        try:
            self.current_bs_result = analyze_ball_screw(
                self.current_recipe, signal_source=signal_source, material=material)
            self._update_bs_tab()
        except Exception as e:
            QMessageBox.critical(self, "Ball Screw Analysis Error", str(e))

    def _on_bs_filter_changed(self):
        """Toggle stabilization point display."""
        if self.current_bs_result:
            self._update_bs_tab()

    def _clear_bs_tab(self):
        """Reset Ball Screw tab to empty state."""
        self.bs_verdict_label.setText("—")
        self.bs_verdict_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; border: 2px solid #45475a;"
            "border-radius: 6px; padding: 4px; color: #a6adc8;")
        self.bs_table.clear()
        self.bs_table.setRowCount(0)
        self.bs_table.setColumnCount(0)
        for canvas in (self.bs_bar_canvas, self.bs_heatmap_canvas):
            old = canvas.figure
            new_fig = Figure(figsize=old.get_size_inches())
            plt.close(old)
            self._update_canvas(canvas, new_fig)

    def _update_bs_tab(self):
        """Refresh all Ball Screw tab visuals from current_bs_result."""
        if not self.current_bs_result:
            return
        bs = self.current_bs_result
        include_stab = self.bs_show_stab_check.isChecked()

        # ── Verdict badge ────────────────────────────────────────────────────
        if bs.overall_pass:
            self.bs_verdict_label.setText("PASS")
            self.bs_verdict_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; border: 2px solid #a6e3a1;"
                "border-radius: 6px; padding: 4px; color: #a6e3a1;")
        else:
            self.bs_verdict_label.setText("FAIL")
            self.bs_verdict_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; border: 2px solid #f38ba8;"
                "border-radius: 6px; padding: 4px; color: #f38ba8;")

        positions, rep_labels, dishing_matrix = get_dishing_matrix(
            bs, include_stabilization=include_stab)
        spec_limit = bs.spec_limit
        n_pos = len(positions)
        n_rep = len(rep_labels)

        # ── Bar chart ────────────────────────────────────────────────────────
        bar_fig = Figure(figsize=(8, 4), facecolor="#1e1e2e")
        ax = bar_fig.add_subplot(111, facecolor="#181825")

        x_pos = np.arange(n_pos)
        bar_w = 0.6
        colors_rep = plt.cm.tab10(np.linspace(0, 0.9, max(n_rep, 1)))

        # Scatter individual repeat values
        for rep_i in range(n_rep):
            vals = dishing_matrix[:, rep_i]
            ax.scatter(x_pos, vals, color=colors_rep[rep_i], s=40, zorder=5,
                       label=rep_labels[rep_i], alpha=0.85)

        # Mean bars
        means = np.nanmean(dishing_matrix, axis=1)
        bar_colors = ["#f38ba8" if v > spec_limit else "#89b4fa" for v in means]
        ax.bar(x_pos, means, width=bar_w, color=bar_colors, alpha=0.35, zorder=3)

        # Spec line
        ax.axhline(spec_limit, color="#f38ba8", linewidth=1.5, linestyle="--",
                   label=f"Spec ({spec_limit} nm)")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(positions, rotation=25, ha="right",
                           color="#cdd6f4", fontsize=9)
        ax.set_ylabel("Dishing (nm)", color="#cdd6f4", fontsize=10)
        ax.set_title(f"Ball Screw Pitch — Dishing per Position [{bs.material}, ≤{spec_limit} nm]",
                     color="#89b4fa", fontsize=11, pad=8)
        ax.tick_params(colors="#cdd6f4", labelsize=9)
        ax.spines[:].set_color("#45475a")
        legend = ax.legend(loc="upper right", fontsize=8,
                           facecolor="#313244", edgecolor="#45475a",
                           labelcolor="#cdd6f4", framealpha=0.85)
        ax.grid(axis="y", color="#313244", linewidth=0.5)
        bar_fig.tight_layout(pad=1.2)
        self._update_canvas(self.bs_bar_canvas, bar_fig)

        # ── Heatmap ──────────────────────────────────────────────────────────
        hm_fig = Figure(figsize=(6, 4), facecolor="#1e1e2e")
        ax2 = hm_fig.add_subplot(111, facecolor="#181825")

        import matplotlib.colors as mcolors
        vmax = max(float(np.nanmax(dishing_matrix)), spec_limit * 1.1)
        vmin = 0.0
        cmap = plt.cm.RdYlGn_r
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        im = ax2.imshow(dishing_matrix.T, aspect="auto", cmap=cmap, norm=norm,
                        origin="upper")
        ax2.set_xticks(range(n_pos))
        ax2.set_xticklabels(positions, rotation=30, ha="right",
                            color="#cdd6f4", fontsize=8)
        ax2.set_yticks(range(n_rep))
        ax2.set_yticklabels(rep_labels, color="#cdd6f4", fontsize=8)
        ax2.tick_params(colors="#cdd6f4")
        ax2.set_title("Dishing Heatmap\n(Position × Repeat)",
                      color="#89b4fa", fontsize=10, pad=6)
        ax2.spines[:].set_color("#45475a")

        # Annotate values + highlight spec failure
        for pos_i in range(n_pos):
            for rep_i in range(n_rep):
                val = dishing_matrix[pos_i, rep_i]
                if not np.isnan(val):
                    txt_color = "white" if val > spec_limit * 0.7 else "black"
                    ax2.text(pos_i, rep_i, f"{val:.2f}",
                             ha="center", va="center",
                             color=txt_color, fontsize=7.5, fontweight="bold")
                    if val > spec_limit:
                        ax2.add_patch(plt.Rectangle(
                            (pos_i - 0.5, rep_i - 0.5), 1, 1,
                            fill=False, edgecolor="#f38ba8", linewidth=2))

        cb = hm_fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="#cdd6f4")
        cb.ax.tick_params(labelcolor="#cdd6f4", labelsize=8)
        cb.set_label("Dishing (nm)", color="#cdd6f4", fontsize=9)
        hm_fig.tight_layout(pad=1.2)
        self._update_canvas(self.bs_heatmap_canvas, hm_fig)

        # ── Summary Table ────────────────────────────────────────────────────
        stat_cols = ["Position"] + rep_labels + ["Mean", "Stdev", "Max", "Spec"]
        self.bs_table.setColumnCount(len(stat_cols))
        self.bs_table.setHorizontalHeaderLabels(stat_cols)
        self.bs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.bs_table.setRowCount(n_pos)
        for row_i, pos in enumerate(positions):
            vals = dishing_matrix[row_i, :]
            valid_vals = vals[~np.isnan(vals)]
            mean_v = float(np.mean(valid_vals)) if len(valid_vals) else float("nan")
            std_v = float(np.std(valid_vals, ddof=0)) if len(valid_vals) else float("nan")
            max_v = float(np.max(valid_vals)) if len(valid_vals) else float("nan")
            is_stab = pos == "1_LT_stab"
            spec_txt = "N/A" if is_stab else ("PASS" if max_v <= spec_limit else "FAIL")

            row_data = [pos]
            for rep_i in range(n_rep):
                v = vals[rep_i]
                row_data.append(f"{v:.3f}" if not np.isnan(v) else "—")
            row_data += [
                f"{mean_v:.3f}" if not np.isnan(mean_v) else "—",
                f"{std_v:.3f}" if not np.isnan(std_v) else "—",
                f"{max_v:.3f}" if not np.isnan(max_v) else "—",
                spec_txt,
            ]

            for col_i, cell_val in enumerate(row_data):
                item = QTableWidgetItem(cell_val)
                item.setTextAlignment(Qt.AlignCenter)
                # Color based on context
                if col_i == 0:  # Position label
                    if is_stab:
                        item.setForeground(QColor("#a6adc8"))
                elif col_i == len(row_data) - 1:  # Spec column
                    if spec_txt == "FAIL":
                        item.setForeground(QColor("#f38ba8"))
                        item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                    elif spec_txt == "PASS":
                        item.setForeground(QColor("#a6e3a1"))
                else:  # Value cells
                    try:
                        fval = float(cell_val)
                        if not is_stab and fval > spec_limit:
                            item.setForeground(QColor("#f38ba8"))
                        elif not is_stab and fval <= spec_limit * 0.8:
                            item.setForeground(QColor("#a6e3a1"))
                    except (ValueError, TypeError):
                        pass
                self.bs_table.setItem(row_i, col_i, item)

    # ─── Flatten ─────────────────────────────────────────────

    def _on_flatten_execute(self):
        if not self.current_recipe:
            QMessageBox.warning(self, "Warning", "Load data first.")
            return

        pos = self.flat_pos_combo.currentText()
        rep_idx = self.flat_rep_combo.currentIndex()
        order = int(self.flat_order_combo.currentText())
        edge_pct = self.flat_edge_spin.value()

        if rep_idx < 0 or rep_idx >= len(self.current_recipe.repeats):
            return

        repeat = self.current_recipe.repeats[rep_idx]
        if pos not in repeat.profiles:
            QMessageBox.warning(self, "Warning",
                                f"No profile for {pos} in Repeat {rep_idx+1}")
            return

        profile = repeat.profiles[pos]
        result = self.flatten_proc.flatten(
            profile.z_nm, profile.x_mm, order=order, edge_percent=edge_pct
        )

        self.flat_stats_label.setText(
            f"Order {order} | OPM: {result.opm_before:.3f} → {result.opm_after:.3f} nm | "
            f"RMS: {result.rms_before:.3f} → {result.rms_after:.3f} nm | Edge: {edge_pct}%"
        )

        fig = create_flatten_preview_figure(result, profile.x_mm, figsize=(10, 8))
        self._update_canvas(self.flatten_canvas, fig)
        self.flat_undo_btn.setEnabled(self.flatten_proc.can_undo)

    def _on_flatten_undo(self):
        prev = self.flatten_proc.undo()
        if prev and self.current_recipe:
            pos = self.flat_pos_combo.currentText()
            rep_idx = self.flat_rep_combo.currentIndex()
            if 0 <= rep_idx < len(self.current_recipe.repeats):
                profile = self.current_recipe.repeats[rep_idx].profiles.get(pos)
                if profile:
                    fig = create_flatten_preview_figure(prev, profile.x_mm)
                    self._update_canvas(self.flatten_canvas, fig)
                    self.flat_stats_label.setText(
                        f"Undo → Order {prev.order} | OPM: {prev.opm_after:.3f} nm")
        self.flat_undo_btn.setEnabled(self.flatten_proc.can_undo)

    # ─── Export ──────────────────────────────────────────────

    def _on_export(self):
        if not self.current_result or not self.current_recipe:
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return

        base = Path(folder)
        rl = self.current_result.range_label
        try:
            export_summary_csv(self.current_result, base / f"summary_{rl}.csv")
            export_avg_line_csv(self.current_recipe, base / f"avg_lines_{rl}.csv")
            export_checklist(self.current_result, base / f"checklist_{rl}.txt")

            for name, canvas in [("profiles", self.profile_canvas),
                                  ("trend", self.trend_canvas),
                                  ("wafer_map", self.wafer_canvas),
                                  ("best5", self.best5_canvas)]:
                canvas.figure.savefig(str(base / f"{name}_{rl}.png"), dpi=150,
                                      facecolor="#1e1e2e", bbox_inches="tight")

            # Ball Screw export (only if analysis has been run)
            if self.current_bs_result:
                include_stab = self.bs_show_stab_check.isChecked()
                export_ball_screw_csv(self.current_bs_result, base, include_stab)
                # Save chart images
                for name, canvas in [("bs_bar", self.bs_bar_canvas),
                                     ("bs_heatmap", self.bs_heatmap_canvas)]:
                    canvas.figure.savefig(str(base / f"{name}_{rl}.png"), dpi=150,
                                          facecolor="#1e1e2e", bbox_inches="tight")

            QMessageBox.information(self, "Export", f"Exported to:\n{folder}")
            self.statusBar().showMessage(f"Exported {rl} to {folder}")
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
