"""Analysis widget — port of RootKinematicsViewer as an embeddable QWidget."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .ki_data_loader import (
    Profile,
    group_key_environmental,
    group_key_genetic,
    load_environmental_mode,
    load_genetic_mode,
)
from .ki_models import LogisticParams, logistic_regr, logistic_velocity
from .ki_outlier_filter import OutlierSettings
from .ki_scaling import ScopeType, UM_PER_PX_KINEMATIC, UM_PER_PX_PLANT


@dataclass
class LogisticMapping:
    v0: Optional[str] = None
    L: Optional[str] = None
    k: Optional[str] = None
    x0: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {"v0": self.v0, "L": self.L, "k": self.k, "x0": self.x0}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "LogisticMapping":
        return cls(v0=data.get("v0"), L=data.get("L"), k=data.get("k"), x0=data.get("x0"))


class MplCanvas(FigureCanvas):
    def __init__(self, parent: Optional[QWidget] = None):
        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class AnalysisWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.mode: str = "genetic"
        self.wt_folder: Optional[Path] = None
        self.mutant_folder: Optional[Path] = None
        self.env_folder: Optional[Path] = None

        self.scope_override: ScopeType = ScopeType.AUTOMATIC

        self.profiles: List[Profile] = []
        self._raw_artist_to_profile: Dict[int, Profile] = {}

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)

        self.logistic_mapping = LogisticMapping()

        self._init_ui()
        self._load_mapping_config()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar-style button row
        btn_row = QHBoxLayout()
        btn_load = QPushButton("Load data")
        btn_load.clicked.connect(self.on_load_clicked)
        btn_param = QPushButton("Parameter mapping\u2026")
        btn_param.clicked.connect(self.on_edit_mapping)
        btn_export = QPushButton("Export summary CSV\u2026")
        btn_export.clicked.connect(self.on_export_summary)
        btn_row.addWidget(btn_load)
        btn_row.addWidget(btn_param)
        btn_row.addWidget(btn_export)
        btn_row.addStretch(1)
        main_layout.addLayout(btn_row)

        self.tabs = QTabWidget()

        self.data_tab = self._build_data_tab()
        self.tabs.addTab(self.data_tab, "\U0001F4C2 Data && Files")

        self.raw_tab = self._build_raw_tab()
        self.model_tab = self._build_model_tab()
        self.common_tab = self._build_common_tab()
        self.stats_tab = self._build_statistics_tab()
        self.tabs.addTab(self.raw_tab, "Raw Overlays")
        self.tabs.addTab(self.model_tab, "Average Comparisons")
        self.tabs.addTab(self.common_tab, "Common Graph Types")
        self.tabs.addTab(self.stats_tab, "Statistical Analysis")

        main_layout.addWidget(self.tabs, 1)

    def _build_data_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        top_row = QHBoxLayout()
        top_row.addWidget(self._build_mode_box())
        top_row.addWidget(self._build_scope_box())
        top_row.addWidget(self._build_outlier_box())
        top_row.addWidget(self._build_time_settings_box())
        layout.addLayout(top_row)

        mid_splitter = QSplitter(Qt.Horizontal)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Filename", "Condition", "Time (hrs)", "Scope", "Full path"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        mid_splitter.addWidget(self.table)

        right_col = QWidget()
        rc_layout = QVBoxLayout(right_col)
        rc_layout.setContentsMargins(0, 0, 0, 0)
        rc_layout.addWidget(self._build_bulk_box())

        log_group = QGroupBox("Details / Log")
        lg_layout = QVBoxLayout(log_group)
        lg_layout.addWidget(self.log_widget)
        rc_layout.addWidget(log_group, 1)

        mid_splitter.addWidget(right_col)
        mid_splitter.setStretchFactor(0, 3)
        mid_splitter.setStretchFactor(1, 2)

        layout.addWidget(mid_splitter, 1)
        return tab

    def _build_mode_box(self) -> QWidget:
        box = QGroupBox("Data input mode")
        layout = QGridLayout(box)

        self.radio_genetic = QRadioButton("Genetic perturbation (WT vs Mutant)")
        self.radio_env = QRadioButton("Environmental perturbation")
        self.radio_genetic.setChecked(True)

        self.mode_group = QButtonGroup(box)
        self.mode_group.addButton(self.radio_genetic)
        self.mode_group.addButton(self.radio_env)
        self.radio_genetic.toggled.connect(self.on_mode_changed)

        layout.addWidget(self.radio_genetic, 0, 0, 1, 3)
        layout.addWidget(self.radio_env, 1, 0, 1, 3)

        self.btn_wt = QPushButton("Select WT folder\u2026")
        self.btn_mutant = QPushButton("Select Mutant folder\u2026")
        self.btn_env = QPushButton("Select data folder\u2026")

        self.lbl_wt = QLabel("WT: (none)")
        self.lbl_mutant = QLabel("Mutant: (none)")
        self.lbl_env = QLabel("Folder: (none)")

        self.btn_wt.clicked.connect(self.on_pick_wt)
        self.btn_mutant.clicked.connect(self.on_pick_mutant)
        self.btn_env.clicked.connect(self.on_pick_env)

        layout.addWidget(self.btn_wt, 2, 0)
        layout.addWidget(self.lbl_wt, 2, 1, 1, 2)
        layout.addWidget(self.btn_mutant, 3, 0)
        layout.addWidget(self.lbl_mutant, 3, 1, 1, 2)
        layout.addWidget(self.btn_env, 4, 0)
        layout.addWidget(self.lbl_env, 4, 1, 1, 2)

        btn_load = QPushButton("Load / Reload CSVs")
        btn_load.clicked.connect(self.on_load_clicked)
        layout.addWidget(btn_load, 5, 0, 1, 3)

        self._update_mode_widgets()
        return box

    def _build_scope_box(self) -> QWidget:
        box = QGroupBox("Scope type & scaling")
        layout = QHBoxLayout(box)

        layout.addWidget(QLabel("Scope type:"))
        self.combo_scope = QComboBox()
        self.combo_scope.addItems([s.value for s in ScopeType])
        self.combo_scope.currentIndexChanged.connect(self.on_scope_changed)
        layout.addWidget(self.combo_scope)

        layout.addWidget(QLabel("Kinematic \u00b5m/px:"))
        self.edit_kine = QLineEdit(str(UM_PER_PX_KINEMATIC))
        self.edit_kine.setMaximumWidth(80)
        layout.addWidget(self.edit_kine)

        layout.addWidget(QLabel("Plant \u00b5m/px:"))
        self.edit_plant = QLineEdit(str(UM_PER_PX_PLANT))
        self.edit_plant.setMaximumWidth(80)
        layout.addWidget(self.edit_plant)

        btn_apply = QPushButton("Apply scaling")
        btn_apply.clicked.connect(self.on_apply_scaling_clicked)
        layout.addWidget(btn_apply)

        layout.addStretch(1)
        return box

    def _build_outlier_box(self) -> QWidget:
        box = QGroupBox("Outlier filter")
        layout = QHBoxLayout(box)

        self.check_outlier_enabled = QCheckBox("Enable")
        self.check_outlier_enabled.setChecked(True)
        self.check_outlier_enabled.toggled.connect(self._on_outlier_toggle)
        layout.addWidget(self.check_outlier_enabled)

        layout.addWidget(QLabel("Method:"))
        self.combo_outlier_method = QComboBox()
        self.combo_outlier_method.addItems(["MAD", "IQR"])
        layout.addWidget(self.combo_outlier_method)

        self.lbl_outlier_thresh = QLabel("Threshold:")
        layout.addWidget(self.lbl_outlier_thresh)
        self.edit_outlier_thresh = QLineEdit("4.0")
        self.edit_outlier_thresh.setMaximumWidth(55)
        self.edit_outlier_thresh.setToolTip(
            "MAD: robust z-score cutoff (default 4.0)\n"
            "IQR: fence multiplier k (default 3.0)"
        )
        layout.addWidget(self.edit_outlier_thresh)

        layout.addStretch(1)
        return box

    def _on_outlier_toggle(self, checked: bool) -> None:
        self.combo_outlier_method.setEnabled(checked)
        self.edit_outlier_thresh.setEnabled(checked)
        self.lbl_outlier_thresh.setEnabled(checked)

    def _get_outlier_settings(self) -> OutlierSettings:
        enabled = self.check_outlier_enabled.isChecked()
        method = self.combo_outlier_method.currentText().lower()
        try:
            thresh = float(self.edit_outlier_thresh.text())
        except ValueError:
            thresh = 4.0 if method == "mad" else 3.0
        return OutlierSettings(
            enabled=enabled,
            method=method,
            mad_thresh=thresh if method == "mad" else 4.0,
            iqr_k=thresh if method == "iqr" else 3.0,
        )

    def _build_time_settings_box(self) -> QWidget:
        box = QGroupBox("Time settings")
        layout = QHBoxLayout(box)

        layout.addWidget(QLabel("Interval:"))
        self.edit_time_interval = QLineEdit("0.5")
        self.edit_time_interval.setMaximumWidth(55)
        self.edit_time_interval.setToolTip(
            "Duration between consecutive imaging sessions (hours).\n"
            "E.g. 0.5 = 30 min between each time point."
        )
        layout.addWidget(self.edit_time_interval)
        layout.addWidget(QLabel("hrs"))

        layout.addWidget(QLabel("Pre-imaging:"))
        self.edit_time_offset = QLineEdit("0")
        self.edit_time_offset.setMaximumWidth(55)
        self.edit_time_offset.setToolTip(
            "Elapsed time before the first imaging session (hours).\n"
            "This offset is added to all time points."
        )
        layout.addWidget(self.edit_time_offset)
        layout.addWidget(QLabel("hrs"))

        layout.addStretch(1)
        return box

    def _build_raw_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Type:"))
        self.combo_raw_type = QComboBox()
        self.combo_raw_type.addItems(["Velocity", "REGR"])
        row1.addWidget(self.combo_raw_type)

        self.check_interp = QCheckBox("Interpolate")
        row1.addWidget(self.check_interp)

        btn_plot = QPushButton("Plot")
        btn_plot.clicked.connect(self.on_plot_raw)
        row1.addWidget(btn_plot)

        self.btn_crop = QPushButton("Crop outliers")
        self.btn_crop.setCheckable(True)
        self.btn_crop.setChecked(False)
        self.btn_crop.setToolTip(
            "Toggle on, then click a data point on the plot to remove that profile."
        )
        row1.addWidget(self.btn_crop)

        btn_save_png = QPushButton("PNG")
        btn_save_png.clicked.connect(lambda: self.save_current_figure("png"))
        row1.addWidget(btn_save_png)

        btn_save_svg = QPushButton("SVG")
        btn_save_svg.clicked.connect(lambda: self.save_current_figure("svg"))
        row1.addWidget(btn_save_svg)

        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()

        self._condition_checks: Dict[str, QCheckBox] = {}
        self._time_checks: Dict[str, QCheckBox] = {}

        row2.addWidget(QLabel("Conditions:"))
        self._cond_checks_layout = QHBoxLayout()
        self._cond_checks_layout.setContentsMargins(0, 0, 0, 0)
        row2.addLayout(self._cond_checks_layout)

        row2.addWidget(QLabel("  Time groups:"))
        self._time_checks_layout = QHBoxLayout()
        self._time_checks_layout.setContentsMargins(0, 0, 0, 0)
        row2.addLayout(self._time_checks_layout)

        row2.addStretch(1)
        layout.addLayout(row2)

        self.canvas_raw = MplCanvas(tab)
        self.canvas_raw.mpl_connect("pick_event", self._on_raw_plot_pick)
        layout.addWidget(self.canvas_raw, 1)
        return tab

    def _refresh_profile_visibility(self) -> None:
        for cb in self._condition_checks.values():
            self._cond_checks_layout.removeWidget(cb)
            cb.deleteLater()
        self._condition_checks.clear()

        for cb in self._time_checks.values():
            self._time_checks_layout.removeWidget(cb)
            cb.deleteLater()
        self._time_checks.clear()

        conditions: set = set()
        time_groups: set = set()
        for p in self.profiles:
            conditions.add(p.condition or "Unknown")
            if p.time_min is not None:
                time_groups.add(self._format_time_label(p.time_min))
            else:
                time_groups.add("Unknown")

        for cond in sorted(conditions):
            cb = QCheckBox(cond)
            cb.setChecked(True)
            self._condition_checks[cond] = cb
            self._cond_checks_layout.addWidget(cb)

        for tg in sorted(time_groups):
            cb = QCheckBox(tg)
            cb.setChecked(True)
            self._time_checks[tg] = cb
            self._time_checks_layout.addWidget(cb)

        self.log(f"Visibility filters: conditions={sorted(conditions)}, times={sorted(time_groups)}")

        if hasattr(self, "combo_common_time"):
            self._refresh_common_combos()
        self._refresh_stat_combos()

    def _build_model_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Reconstruct:"))
        self.combo_model_type = QComboBox()
        self.combo_model_type.addItems(["Velocity", "REGR"])
        controls.addWidget(self.combo_model_type)

        controls.addWidget(QLabel("Time:"))
        self.combo_model_time = QComboBox()
        self.combo_model_time.addItem("All")
        controls.addWidget(self.combo_model_time)

        self.check_sem = QCheckBox("Use SEM (instead of SD)")
        controls.addWidget(self.check_sem)

        btn_plot = QPushButton("Plot average comparison")
        btn_plot.clicked.connect(self.on_plot_model)
        controls.addWidget(btn_plot)

        btn_save_png = QPushButton("Save PNG\u2026")
        btn_save_png.clicked.connect(lambda: self.save_current_figure("png", model=True))
        controls.addWidget(btn_save_png)

        btn_save_svg = QPushButton("Save SVG\u2026")
        btn_save_svg.clicked.connect(lambda: self.save_current_figure("svg", model=True))
        controls.addWidget(btn_save_svg)

        btn_export = QPushButton("Export CSV\u2026")
        btn_export.setToolTip(
            "Export parameter summary and curve data (param-based + pointwise) to CSV"
        )
        btn_export.clicked.connect(self.on_export_summary)
        controls.addWidget(btn_export)

        controls.addStretch(1)
        layout.addLayout(controls)

        self.canvas_model = MplCanvas(tab)
        layout.addWidget(self.canvas_model, 1)
        return tab

    def _build_common_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Graph type:"))
        self.combo_common_mode = QComboBox()
        self.combo_common_mode.addItems(["Scope Comparison", "REGR over Time"])
        self.combo_common_mode.currentIndexChanged.connect(self._on_common_mode_changed)
        row1.addWidget(self.combo_common_mode)

        row1.addWidget(QLabel("Metric:"))
        self.combo_common_metric = QComboBox()
        self.combo_common_metric.addItems(["REGR", "Velocity"])
        row1.addWidget(self.combo_common_metric)

        self.check_common_sem = QCheckBox("Use SEM")
        row1.addWidget(self.check_common_sem)

        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()

        self._scope_cmp_widget = QWidget()
        sc_layout = QHBoxLayout(self._scope_cmp_widget)
        sc_layout.setContentsMargins(0, 0, 0, 0)
        sc_layout.addWidget(QLabel("Time point:"))
        self.combo_common_time = QComboBox()
        self.combo_common_time.addItem("All")
        sc_layout.addWidget(self.combo_common_time)
        row2.addWidget(self._scope_cmp_widget)

        self._time_series_widget = QWidget()
        ts_layout = QHBoxLayout(self._time_series_widget)
        ts_layout.setContentsMargins(0, 0, 0, 0)
        ts_layout.addWidget(QLabel("Condition:"))
        self.combo_common_condition = QComboBox()
        self.combo_common_condition.addItem("All")
        ts_layout.addWidget(self.combo_common_condition)
        self._time_series_widget.setVisible(False)
        row2.addWidget(self._time_series_widget)

        btn_plot = QPushButton("Plot")
        btn_plot.clicked.connect(self.on_plot_common)
        row2.addWidget(btn_plot)

        btn_save_png = QPushButton("Save PNG\u2026")
        btn_save_png.clicked.connect(lambda: self.save_current_figure("png", canvas_override=self.canvas_common))
        row2.addWidget(btn_save_png)

        btn_save_svg = QPushButton("Save SVG\u2026")
        btn_save_svg.clicked.connect(lambda: self.save_current_figure("svg", canvas_override=self.canvas_common))
        row2.addWidget(btn_save_svg)

        row2.addStretch(1)
        layout.addLayout(row2)

        self.canvas_common = MplCanvas(tab)
        layout.addWidget(self.canvas_common, 1)
        return tab

    def _build_statistics_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        row1 = QHBoxLayout()

        row1.addWidget(QLabel("Compare by:"))
        self.combo_stat_mode = QComboBox()
        self.combo_stat_mode.addItems(["Condition", "Scope", "Time Point"])
        self.combo_stat_mode.currentIndexChanged.connect(self._on_stat_mode_changed)
        row1.addWidget(self.combo_stat_mode)

        row1.addWidget(QLabel("Metric:"))
        self.combo_stat_metric = QComboBox()
        self.combo_stat_metric.addItems([
            "Max Velocity", "Max REGR", "REGR Peak Position", "REGR AUC",
        ])
        row1.addWidget(self.combo_stat_metric)

        row1.addWidget(QLabel("Plot:"))
        self.combo_stat_plot = QComboBox()
        self.combo_stat_plot.addItems(["Bar + Points", "Box Plot", "Trajectory"])
        self.combo_stat_plot.currentIndexChanged.connect(self._on_stat_plot_type_changed)
        row1.addWidget(self.combo_stat_plot)

        row1.addWidget(QLabel("Test:"))
        self.combo_stat_test = QComboBox()
        self.combo_stat_test.addItems([
            "t-test (Welch)", "Mann-Whitney U",
            "One-way ANOVA", "Kruskal-Wallis",
        ])
        row1.addWidget(self.combo_stat_test)

        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()

        self.lbl_stat_time = QLabel("Time:")
        row2.addWidget(self.lbl_stat_time)
        self.combo_stat_time = QComboBox()
        self.combo_stat_time.addItem("All")
        row2.addWidget(self.combo_stat_time)

        self.lbl_stat_cond = QLabel("Condition:")
        row2.addWidget(self.lbl_stat_cond)
        self.combo_stat_cond = QComboBox()
        self.combo_stat_cond.addItem("All")
        row2.addWidget(self.combo_stat_cond)

        self.lbl_stat_scope = QLabel("Scope:")
        row2.addWidget(self.lbl_stat_scope)
        self.combo_stat_scope = QComboBox()
        self.combo_stat_scope.addItems(["All", "Kinematic", "Plant"])
        row2.addWidget(self.combo_stat_scope)

        btn_analyze = QPushButton("Analyze")
        btn_analyze.clicked.connect(self._on_plot_statistics)
        row2.addWidget(btn_analyze)

        btn_save_png = QPushButton("Save PNG\u2026")
        btn_save_png.clicked.connect(
            lambda: self.save_current_figure("png", canvas_override=self.canvas_stats)
        )
        row2.addWidget(btn_save_png)

        btn_save_svg = QPushButton("Save SVG\u2026")
        btn_save_svg.clicked.connect(
            lambda: self.save_current_figure("svg", canvas_override=self.canvas_stats)
        )
        row2.addWidget(btn_save_svg)

        row2.addStretch(1)
        layout.addLayout(row2)

        from PySide6.QtCore import Qt as _Qt
        stat_splitter = QSplitter(_Qt.Vertical)

        self.canvas_stats = MplCanvas(tab)
        self.canvas_stats.setMinimumHeight(300)
        stat_splitter.addWidget(self.canvas_stats)

        self.stats_results = QTextEdit()
        self.stats_results.setReadOnly(True)
        self.stats_results.setStyleSheet(
            "QTextEdit { font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; }"
        )
        stat_splitter.addWidget(self.stats_results)

        stat_splitter.setStretchFactor(0, 4)
        stat_splitter.setStretchFactor(1, 1)
        stat_splitter.setSizes([500, 120])

        layout.addWidget(stat_splitter, 1)

        self._on_stat_mode_changed()
        return tab

    def _on_stat_mode_changed(self, _index: int = 0) -> None:
        plot_type = self.combo_stat_plot.currentText()
        if plot_type == "Trajectory":
            self.lbl_stat_cond.setVisible(False)
            self.combo_stat_cond.setVisible(False)
            self.lbl_stat_time.setVisible(False)
            self.combo_stat_time.setVisible(False)
            self.lbl_stat_scope.setVisible(True)
            self.combo_stat_scope.setVisible(True)
            self.combo_stat_mode.setEnabled(False)
            self.combo_stat_test.setEnabled(True)
        else:
            self.combo_stat_mode.setEnabled(True)
            self.combo_stat_test.setEnabled(True)
            mode = self.combo_stat_mode.currentText()
            self.lbl_stat_cond.setVisible(mode != "Condition")
            self.combo_stat_cond.setVisible(mode != "Condition")
            self.lbl_stat_scope.setVisible(mode != "Scope")
            self.combo_stat_scope.setVisible(mode != "Scope")
            self.lbl_stat_time.setVisible(mode != "Time Point")
            self.combo_stat_time.setVisible(mode != "Time Point")

    def _on_stat_plot_type_changed(self, _index: int = 0) -> None:
        self._on_stat_mode_changed()

    def _raw_time_to_hours(self, raw_t: float) -> float:
        try:
            interval = float(self.edit_time_interval.text())
        except (ValueError, AttributeError):
            interval = 0.5
        try:
            offset = float(self.edit_time_offset.text())
        except (ValueError, AttributeError):
            offset = 0.0
        return offset + raw_t * interval

    def _format_time_label(self, raw_t: float) -> str:
        hrs = self._raw_time_to_hours(raw_t)
        if hrs == int(hrs):
            return f"{int(hrs)} hr" if int(hrs) != 1 else "1 hr"
        return f"{hrs:g} hrs"

    def _refresh_stat_combos(self) -> None:
        if not hasattr(self, "combo_stat_time"):
            return
        times: set[str] = set()
        conditions: set[str] = set()
        for p in self.profiles:
            if p.time_min is not None:
                times.add(self._format_time_label(p.time_min))
            else:
                times.add("Unknown")
            conditions.add(p.condition or "Unknown")

        self.combo_stat_time.clear()
        self.combo_stat_time.addItem("All")
        for t in sorted(times):
            self.combo_stat_time.addItem(t)

        self.combo_stat_cond.clear()
        self.combo_stat_cond.addItem("All")
        for c in sorted(conditions):
            self.combo_stat_cond.addItem(c)

    def _on_common_mode_changed(self, index: int) -> None:
        self._scope_cmp_widget.setVisible(index == 0)
        self._time_series_widget.setVisible(index == 1)

    def _refresh_common_combos(self) -> None:
        self.combo_common_time.clear()
        self.combo_common_time.addItem("All")
        times: set = set()
        conditions: set = set()
        for p in self.profiles:
            if p.time_min is not None:
                times.add(self._format_time_label(p.time_min))
            else:
                times.add("Unknown")
            conditions.add(p.condition or "Unknown")
        for t in sorted(times):
            self.combo_common_time.addItem(t)

        self.combo_common_condition.clear()
        self.combo_common_condition.addItem("All")
        for c in sorted(conditions):
            self.combo_common_condition.addItem(c)

    def _build_bulk_box(self) -> QWidget:
        box = QGroupBox("Bulk helpers")
        layout = QGridLayout(box)

        layout.addWidget(QLabel("Condition:"), 0, 0)
        btn_ctrl = QPushButton("Control")
        btn_ctrl.clicked.connect(lambda: self.bulk_set_condition("Control"))
        layout.addWidget(btn_ctrl, 0, 1)

        btn_wt = QPushButton("WT")
        btn_wt.clicked.connect(lambda: self.bulk_set_condition("WT"))
        layout.addWidget(btn_wt, 0, 2)

        btn_mut = QPushButton("Mutant")
        btn_mut.clicked.connect(lambda: self.bulk_set_condition("Mutant"))
        layout.addWidget(btn_mut, 0, 3)

        btn_drought = QPushButton("Drought")
        btn_drought.clicked.connect(lambda: self.bulk_set_condition("Drought"))
        layout.addWidget(btn_drought, 0, 4)

        layout.addWidget(QLabel("Scope:"), 1, 0)
        btn_kine = QPushButton("Kinematic")
        btn_kine.clicked.connect(lambda: self.bulk_set_scope("Kinematic"))
        layout.addWidget(btn_kine, 1, 1)

        btn_plant = QPushButton("Plant")
        btn_plant.clicked.connect(lambda: self.bulk_set_scope("Plant"))
        layout.addWidget(btn_plant, 1, 2)

        btn_time = QPushButton("Set time\u2026")
        btn_time.clicked.connect(self.bulk_set_time)
        layout.addWidget(btn_time, 1, 3)

        btn_autoinfer = QPushButton("Auto-infer all")
        btn_autoinfer.clicked.connect(self.bulk_auto_infer)
        layout.addWidget(btn_autoinfer, 1, 4)

        return box

    # ------------------------------------------------------------------
    # Mode & folder selection
    # ------------------------------------------------------------------
    def on_mode_changed(self) -> None:
        self.mode = "genetic" if self.radio_genetic.isChecked() else "environmental"
        self._update_mode_widgets()

    def _update_mode_widgets(self) -> None:
        genetic = self.mode == "genetic"
        self.btn_wt.setEnabled(genetic)
        self.btn_mutant.setEnabled(genetic)
        self.lbl_wt.setEnabled(genetic)
        self.lbl_mutant.setEnabled(genetic)
        self.btn_env.setEnabled(not genetic)
        self.lbl_env.setEnabled(not genetic)

    def on_pick_wt(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select WT folder")
        if folder:
            self.wt_folder = Path(folder)
            self.lbl_wt.setText(str(self.wt_folder))

    def on_pick_mutant(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Mutant folder")
        if folder:
            self.mutant_folder = Path(folder)
            self.lbl_mutant.setText(str(self.mutant_folder))

    def on_pick_env(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select data folder")
        if folder:
            self.env_folder = Path(folder)
            self.lbl_env.setText(str(self.env_folder))

    # ------------------------------------------------------------------
    # Scope / scaling
    # ------------------------------------------------------------------
    def on_scope_changed(self) -> None:
        text = self.combo_scope.currentText()
        for s in ScopeType:
            if s.value == text:
                self.scope_override = s
                break

    def on_apply_scaling_clicked(self) -> None:
        from . import ki_scaling as _ki_scaling
        try:
            kine = float(self.edit_kine.text())
            plant = float(self.edit_plant.text())
        except ValueError:
            QMessageBox.warning(self, "RootKinematicsViewer", "Scaling values must be numeric.")
            return
        _ki_scaling.UM_PER_PX_KINEMATIC = kine
        _ki_scaling.UM_PER_PX_PLANT = plant
        self.log("Updated scaling constants. Reload data to apply to profiles.")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def on_load_clicked(self) -> None:
        try:
            self._load_profiles()
        except Exception as exc:
            self._show_error("Failed to load data", str(exc))

    def _load_profiles(self) -> None:
        self.log_widget.clear()
        self.profiles.clear()
        outlier_cfg = self._get_outlier_settings()
        if self.mode == "genetic":
            if self.wt_folder is None:
                self._show_error("Folder required", "Please select a WT folder.")
                return
            self.profiles = load_genetic_mode(
                self.wt_folder,
                self.mutant_folder,
                scope_override=self.scope_override,
                outlier_settings=outlier_cfg,
            )
        else:
            if self.env_folder is None:
                self._show_error("Folder required", "Please select a data folder.")
                return
            self.profiles = load_environmental_mode(
                self.env_folder, scope_override=self.scope_override, auto_infer=True,
                outlier_settings=outlier_cfg,
            )

        if outlier_cfg.enabled:
            n_filtered = sum(
                1 for p in self.profiles
                if any("Outlier filter" in w for w in p.load_warnings)
            )
            self.log(
                f"Outlier filter ({outlier_cfg.method.upper()}, "
                f"thresh={outlier_cfg.mad_thresh if outlier_cfg.method == 'mad' else outlier_cfg.iqr_k}): "
                f"{n_filtered}/{len(self.profiles)} profiles had points removed."
            )
        self.log(f"Loaded {len(self.profiles)} profiles.")
        self._refresh_table()
        self._refresh_time_combos()
        self._refresh_profile_visibility()
        self._refresh_common_combos()
        self._refresh_stat_combos()

    # ------------------------------------------------------------------
    # Table + bulk helpers
    # ------------------------------------------------------------------
    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self.profiles))
        for row, p in enumerate(self.profiles):
            self.table.setItem(row, 0, QTableWidgetItem(p.filename))
            self.table.setItem(row, 1, QTableWidgetItem(p.condition or ""))
            time_text = "" if p.time_min is None else self._format_time_label(p.time_min)
            self.table.setItem(row, 2, QTableWidgetItem(time_text))
            scope_text = p.scope_type.value if p.scope_type else ""
            self.table.setItem(row, 3, QTableWidgetItem(scope_text))
            self.table.setItem(row, 4, QTableWidgetItem(str(p.path)))
        self.table.resizeColumnsToContents()

    def _pull_table_edits(self) -> None:
        for row, p in enumerate(self.profiles):
            cond_item = self.table.item(row, 1)
            time_item = self.table.item(row, 2)
            scope_item = self.table.item(row, 3)

            p.condition = cond_item.text().strip() or None if cond_item else p.condition

            if time_item and time_item.text().strip():
                raw = time_item.text().strip()
                raw = raw.replace("hrs", "").replace("hr", "").strip()
                try:
                    display_hrs = float(raw)
                    try:
                        interval = float(self.edit_time_interval.text())
                    except (ValueError, AttributeError):
                        interval = 0.5
                    try:
                        offset = float(self.edit_time_offset.text())
                    except (ValueError, AttributeError):
                        offset = 0.0
                    p.time_min = (display_hrs - offset) / interval if interval != 0 else 0.0
                except ValueError:
                    p.time_min = None
            scope_text = scope_item.text().strip() if scope_item else ""
            for s in ScopeType:
                if s.value == scope_text:
                    p.scope_type = s
                    break

    def selected_rows(self) -> List[int]:
        rows = set()
        for idx in self.table.selectedIndexes():
            rows.add(idx.row())
        return sorted(rows)

    def bulk_set_condition(self, condition: str) -> None:
        rows = self.selected_rows()
        if not rows:
            return
        for r in rows:
            self.table.setItem(r, 1, QTableWidgetItem(condition))
        self._pull_table_edits()
        self.log(f"Set condition={condition} for {len(rows)} profiles.")
        self._refresh_time_combos()
        self._refresh_profile_visibility()

    def bulk_set_time(self) -> None:
        rows = self.selected_rows()
        if not rows:
            return
        value, ok = QInputDialog.getDouble(
            self, "Set time", "Time (minutes):", 0.0, -1e6, 1e6, 1
        )
        if not ok:
            return
        for r in rows:
            self.table.setItem(r, 2, QTableWidgetItem(f"{value:g}"))
        self._pull_table_edits()
        self.log(f"Set time={value:g} min for {len(rows)} profiles.")
        self._refresh_time_combos()
        self._refresh_profile_visibility()

    def bulk_set_scope(self, scope_label: str) -> None:
        rows = self.selected_rows()
        if not rows:
            return
        for r in rows:
            self.table.setItem(r, 3, QTableWidgetItem(scope_label))
        self._pull_table_edits()
        self.log(f"Set scope={scope_label} for {len(rows)} profiles.")

    def bulk_auto_infer(self) -> None:
        from .ki_data_loader import (
            parse_structured_filename,
            infer_condition_from_name,
            infer_time_from_name,
            infer_scope_from_name,
        )

        rows = self.selected_rows()
        if not rows:
            rows = list(range(len(self.profiles)))

        for r in rows:
            p = self.profiles[r]
            info = parse_structured_filename(p.filename)
            if info is not None:
                if info.condition is not None:
                    p.condition = info.condition
                if info.time_min is not None:
                    p.time_min = info.time_min
                if info.scope is not None:
                    p.scope_type = info.scope
            else:
                t = infer_time_from_name(p.filename)
                if t is not None:
                    p.time_min = t
                cond = infer_condition_from_name(p.filename)
                if cond is not None:
                    p.condition = cond
                scope = infer_scope_from_name(p.filename)
                if scope is not None:
                    p.scope_type = scope

        self._refresh_table()
        self._refresh_time_combos()
        self._refresh_profile_visibility()
        self.log(f"Auto-inferred metadata for {len(rows)} profiles.")

    def _refresh_time_combos(self) -> None:
        self._pull_table_edits()
        model_times: List[str] = ["All"]

        if self.mode == "environmental":
            labels = set()
            for p in self.profiles:
                if p.time_min is None:
                    labels.add("Unknown")
                else:
                    labels.add(self._format_time_label(p.time_min))
            for lbl in sorted(labels):
                model_times.append(lbl)
        self.combo_model_time.clear()
        self.combo_model_time.addItems(model_times)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _filter_profiles_for_raw(self) -> List[Profile]:
        self._pull_table_edits()
        return list(self.profiles)

    def on_plot_raw(self) -> None:
        profiles = self._filter_profiles_for_raw()
        if not profiles:
            self._show_info("No data", "No profiles to plot. Load data first.")
            return

        visible_conds = {k for k, cb in self._condition_checks.items() if cb.isChecked()}
        visible_times = {k for k, cb in self._time_checks.items() if cb.isChecked()}
        has_cond_filter = len(self._condition_checks) > 0
        has_time_filter = len(self._time_checks) > 0
        self.log(f"[debug] cond checkboxes: {list(self._condition_checks.keys())} visible={visible_conds}")
        self.log(f"[debug] time checkboxes: {list(self._time_checks.keys())} visible={visible_times}")
        if profiles:
            sample = profiles[0]
            self.log(f"[debug] sample profile: cond={sample.condition!r}, time={sample.time_min!r}")

        def _passes(p: Profile) -> bool:
            if has_cond_filter:
                cond = p.condition or "Unknown"
                if cond not in visible_conds:
                    return False
            if has_time_filter:
                tg = self._format_time_label(p.time_min) if p.time_min is not None else "Unknown"
                if tg not in visible_times:
                    return False
            return True

        profiles = [p for p in profiles if _passes(p)]
        self.log(f"[debug] after visibility filter: {len(profiles)} profiles remain")
        if not profiles:
            self._show_info("No data", "All groups are hidden. Check the Conditions / Time group checkboxes above the plot.")
            return

        kind = self.combo_raw_type.currentText()
        interp = self.check_interp.isChecked()

        ax = self.canvas_raw.ax
        ax.clear()
        self._raw_artist_to_profile = {}

        xmin: Optional[float] = None
        xmax: Optional[float] = None
        ymin: Optional[float] = None
        ymax: Optional[float] = None

        def _update_bounds(x: np.ndarray, y: np.ndarray) -> None:
            nonlocal xmin, xmax, ymin, ymax
            xf = np.asarray(x, dtype=float).ravel()
            yf = np.asarray(y, dtype=float).ravel()
            n = min(xf.size, yf.size)
            if n == 0:
                return
            xf, yf = xf[:n], yf[:n]
            m = np.isfinite(xf) & np.isfinite(yf)
            if not np.any(m):
                return
            xmin = float(np.min(xf[m])) if xmin is None else min(xmin, float(np.min(xf[m])))
            xmax = float(np.max(xf[m])) if xmax is None else max(xmax, float(np.max(xf[m])))
            ymin = float(np.min(yf[m])) if ymin is None else min(ymin, float(np.min(yf[m])))
            ymax = float(np.max(yf[m])) if ymax is None else max(ymax, float(np.max(yf[m])))

        colors = {
            "WT": "tab:blue", "Control": "tab:blue",
            "Mutant": "tab:orange", "Perturbed": "tab:orange",
            "Drought": "tab:red", "Salt": "tab:green",
            "Treated": "tab:purple", "Unknown": "gray",
        }

        cond_to_profiles: Dict[str, List[Profile]] = {}
        for p in profiles:
            cond = p.condition or "Unknown"
            cond_to_profiles.setdefault(cond, []).append(p)

        legend_added: set = set()

        for cond, plist in cond_to_profiles.items():
            xs_all: List[Tuple[np.ndarray, np.ndarray, Profile]] = []
            color = colors.get(cond, "black")
            for p in plist:
                if p.x_um.size == 0:
                    continue
                if kind == "Velocity":
                    if p.velocity is None:
                        continue
                    y = np.asarray(p.velocity, dtype=float)
                    x = np.asarray(p.x_um, dtype=float)
                else:
                    if p.regr is None:
                        continue
                    y = np.asarray(p.regr, dtype=float)
                    x = np.asarray(p.regr_x, dtype=float) if p.regr_x is not None else np.asarray(p.x_um, dtype=float)

                if interp:
                    xs_all.append((x, y, p))
                    continue

                label = cond if cond not in legend_added else None
                if kind == "REGR" and p.regr_x is not None:
                    art, = ax.plot(x, y, color=color, alpha=0.3, linewidth=1.0, picker=5, label=label)
                else:
                    art = ax.scatter(x, y, color=color, alpha=0.3, s=4, picker=8, label=label)
                if label:
                    legend_added.add(cond)
                self._raw_artist_to_profile[id(art)] = p
                _update_bounds(x, y)

            if interp and xs_all:
                xmin_i = min(float(np.nanmin(xv)) for xv, _, _ in xs_all)
                xmax_i = max(float(np.nanmax(xv)) for xv, _, _ in xs_all)
                grid = np.linspace(xmin_i, xmax_i, 300)
                for x, y, p in xs_all:
                    if x.size < 2:
                        continue
                    yi = np.interp(grid, x, y)
                    label = cond if cond not in legend_added else None
                    if kind == "REGR" and p.regr_x is not None:
                        art, = ax.plot(grid, yi, color=color, alpha=0.3, linewidth=1.0, picker=5, label=label)
                    else:
                        art = ax.scatter(grid, yi, color=color, alpha=0.3, s=4, picker=8, label=label)
                    if label:
                        legend_added.add(cond)
                    self._raw_artist_to_profile[id(art)] = p
                    _update_bounds(grid, yi)

        ax.set_xlabel("Position (\u00b5m)")
        ax.set_ylabel("Velocity (\u00b5m/min)" if kind == "Velocity" else "REGR (1/min)")
        ax.set_title("Raw overlays")
        if legend_added:
            ax.legend(loc="best")

        if xmin is None or xmax is None or ymin is None or ymax is None:
            self._show_info(
                "Nothing to plot",
                "No finite data points were found to plot. Check that the loaded "
                "data contains valid numeric values for position and the selected metric.",
            )
        else:
            xpad = (xmax - xmin) * 0.05 or 1.0
            ypad = (ymax - ymin) * 0.05 or 1.0
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)

        ax.figure.tight_layout()
        self.canvas_raw.draw()

    def _on_raw_plot_pick(self, event) -> None:
        if not self.btn_crop.isChecked():
            return
        try:
            profile = self._raw_artist_to_profile.get(id(event.artist))
        except Exception:
            profile = None
        if profile is None:
            return
        name = profile.filename
        reply = QMessageBox.question(
            self,
            "Remove profile",
            f"Remove \"{name}\" from analysis? It will be removed from the table and future plots.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes and profile in self.profiles:
            self.profiles.remove(profile)
            self._refresh_table()
            self._refresh_time_combos()
            self.on_plot_raw()

    def on_plot_model(self) -> None:
        self._pull_table_edits()
        if not self.profiles:
            self._show_info("No data", "No profiles loaded.")
            return

        kind = self.combo_model_type.currentText()
        time_sel = self.combo_model_time.currentText()
        use_sem = self.check_sem.isChecked()
        ax = self.canvas_model.ax
        ax.clear()

        xmin: Optional[float] = None
        xmax: Optional[float] = None
        ymin: Optional[float] = None
        ymax: Optional[float] = None

        def _update_bounds(x: np.ndarray, y: np.ndarray) -> None:
            nonlocal xmin, xmax, ymin, ymax
            xf = np.asarray(x, dtype=float).ravel()
            yf = np.asarray(y, dtype=float).ravel()
            n = min(xf.size, yf.size)
            if n == 0:
                return
            xf, yf = xf[:n], yf[:n]
            m = np.isfinite(xf) & np.isfinite(yf)
            if not np.any(m):
                return
            xmin = float(np.min(xf[m])) if xmin is None else min(xmin, float(np.min(xf[m])))
            xmax = float(np.max(xf[m])) if xmax is None else max(xmax, float(np.max(xf[m])))
            ymin = float(np.min(yf[m])) if ymin is None else min(ymin, float(np.min(yf[m])))
            ymax = float(np.max(yf[m])) if ymax is None else max(ymax, float(np.max(yf[m])))

        def _logistic3(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return a / (1.0 + np.exp(-b * (x - c)))

        groups: Dict[Tuple, List[Profile]] = {}
        for p in self.profiles:
            if self.mode == "genetic":
                key = group_key_genetic(p)
            else:
                key = group_key_environmental(p)
                if time_sel != "All":
                    cond, tlabel = key
                    if tlabel != time_sel:
                        continue
            groups.setdefault(key, []).append(p)

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

        any_plotted = False
        for idx, (key, plist) in enumerate(sorted(groups.items())):
            param_vecs: List[np.ndarray] = []
            max_x = 0.0
            for p in plist:
                if p.logistic_fit_3p is not None:
                    param_vecs.append(p.logistic_fit_3p)
                    if p.x_um.size:
                        max_x = max(max_x, float(np.nanmax(p.x_um)))
                elif p.logistic_params is not None:
                    lp = p.logistic_params
                    param_vecs.append(np.array([lp.v0, lp.L, lp.k, lp.x0]))
                    if p.x_um.size:
                        max_x = max(max_x, float(np.nanmax(p.x_um)))

            if not param_vecs:
                continue

            arr = np.array(param_vecs, dtype=float)
            m_mean = np.nanmean(arr, axis=0)
            m_sd = np.nanstd(arr, axis=0, ddof=0)
            if use_sem and arr.shape[0] > 1:
                m_sd = m_sd / np.sqrt(arr.shape[0])
            m_plus = m_mean + m_sd
            m_minus = m_mean - m_sd

            x_grid = np.linspace(0.0, max_x if max_x > 0 else 1000.0, 500)

            if arr.shape[1] == 3:
                v_mean = _logistic3(x_grid, *m_mean)
                v_plus = _logistic3(x_grid, *m_plus)
                v_minus = _logistic3(x_grid, *m_minus)
            else:
                from .ki_models import params_from_array
                v_mean = logistic_velocity(x_grid, params_from_array(m_mean))
                v_plus = logistic_velocity(x_grid, params_from_array(m_plus))
                v_minus = logistic_velocity(x_grid, params_from_array(m_minus))

            if kind == "Velocity":
                y_mean = v_mean
                y_plus = v_plus
                y_minus = v_minus
                ylabel = "Velocity (\u00b5m/min)"
            else:
                y_mean = np.gradient(v_mean, x_grid)
                y_plus = np.gradient(v_plus, x_grid)
                y_minus = np.gradient(v_minus, x_grid)
                ylabel = "REGR (1/min)"

            color = colors[idx % len(colors)]
            label = " / ".join(str(k) for k in key)
            n_roots = arr.shape[0]
            spread_label = "SEM" if use_sem else "SD"
            ax.plot(x_grid, y_mean, color=color, linewidth=2.0,
                    label=f"{label} (n={n_roots})")
            ax.fill_between(x_grid, y_minus, y_plus, color=color, alpha=0.2,
                            label=f"{label} \u00b1 {spread_label}")
            _update_bounds(x_grid, y_plus)
            _update_bounds(x_grid, y_minus)
            _update_bounds(x_grid, y_mean)
            any_plotted = True

        if not any_plotted:
            self._show_info(
                "No fit parameters",
                "No logistic fit parameters could be computed for the loaded profiles.\n"
                "Make sure each profile has at least 10 finite data points.",
            )
            self.canvas_model.draw()
            return

        ax.set_xlabel("Position (\u00b5m)")
        ax.set_ylabel(ylabel)
        ax.set_title("Average Comparisons")
        ax.legend(loc="best")

        if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
            xpad = (xmax - xmin) * 0.05 or 1.0
            ypad = (ymax - ymin) * 0.05 or 1.0
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)

        ax.figure.tight_layout()
        self.canvas_model.draw()

    # ------------------------------------------------------------------
    # Common Graph Types
    # ------------------------------------------------------------------
    def _reconstruct_group_curve(
        self,
        plist: List[Profile],
        kind: str,
        use_sem: bool,
    ) -> Optional[Dict[str, np.ndarray]]:
        def _logistic3(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return a / (1.0 + np.exp(-b * (x - c)))

        param_vecs: List[np.ndarray] = []
        max_x = 0.0
        for p in plist:
            if p.logistic_fit_3p is not None:
                param_vecs.append(p.logistic_fit_3p)
                if p.x_um.size:
                    max_x = max(max_x, float(np.nanmax(p.x_um)))
            elif p.logistic_params is not None:
                lp = p.logistic_params
                param_vecs.append(np.array([lp.v0, lp.L, lp.k, lp.x0]))
                if p.x_um.size:
                    max_x = max(max_x, float(np.nanmax(p.x_um)))
        if not param_vecs:
            return None

        arr = np.array(param_vecs, dtype=float)
        m_mean = np.nanmean(arr, axis=0)
        m_sd = np.nanstd(arr, axis=0, ddof=0)
        if use_sem and arr.shape[0] > 1:
            m_sd = m_sd / np.sqrt(arr.shape[0])
        m_plus = m_mean + m_sd
        m_minus = m_mean - m_sd

        x_grid = np.linspace(0.0, max_x if max_x > 0 else 1000.0, 500)

        if arr.shape[1] == 3:
            v_mean = _logistic3(x_grid, *m_mean)
            v_plus = _logistic3(x_grid, *m_plus)
            v_minus = _logistic3(x_grid, *m_minus)
        else:
            from .ki_models import params_from_array
            v_mean = logistic_velocity(x_grid, params_from_array(m_mean))
            v_plus = logistic_velocity(x_grid, params_from_array(m_plus))
            v_minus = logistic_velocity(x_grid, params_from_array(m_minus))

        if kind == "Velocity":
            y_mean, y_plus, y_minus = v_mean, v_plus, v_minus
        else:
            y_mean = np.gradient(v_mean, x_grid)
            y_plus = np.gradient(v_plus, x_grid)
            y_minus = np.gradient(v_minus, x_grid)

        return {
            "x_grid": x_grid,
            "y_mean": y_mean,
            "y_plus": y_plus,
            "y_minus": y_minus,
            "n_roots": arr.shape[0],
        }

    def on_plot_common(self) -> None:
        self._pull_table_edits()
        if not self.profiles:
            self._show_info("No data", "No profiles loaded.")
            return

        mode = self.combo_common_mode.currentText()
        metric = self.combo_common_metric.currentText()
        use_sem = self.check_common_sem.isChecked()
        ax = self.canvas_common.ax
        ax.clear()

        if mode == "Scope Comparison":
            self._plot_scope_comparison(ax, metric, use_sem)
        else:
            self._plot_regr_over_time(ax, metric, use_sem)

        ax.figure.tight_layout()
        self.canvas_common.draw()

    def _plot_scope_comparison(self, ax, metric: str, use_sem: bool) -> None:
        time_sel = self.combo_common_time.currentText()

        groups: Dict[Tuple[str, str], List[Profile]] = {}
        for p in self.profiles:
            if time_sel != "All":
                tlabel = self._format_time_label(p.time_min) if p.time_min is not None else "Unknown"
                if tlabel != time_sel:
                    continue
            cond = p.condition or "Unknown"
            scope = p.scope_type.value if p.scope_type and p.scope_type != ScopeType.AUTOMATIC else "Unknown"
            groups.setdefault((cond, scope), []).append(p)

        if not groups:
            self._show_info("No data", "No profiles match the selected time point.")
            return

        style_map = {
            ("Control", "Kinematic"): {"color": "tab:blue", "linestyle": "-"},
            ("Control", "Plant"):     {"color": "tab:blue", "linestyle": "--"},
            ("Drought", "Kinematic"): {"color": "tab:red", "linestyle": "-"},
            ("Drought", "Plant"):     {"color": "tab:red", "linestyle": "--"},
            ("WT", "Kinematic"):      {"color": "tab:blue", "linestyle": "-"},
            ("WT", "Plant"):          {"color": "tab:blue", "linestyle": "--"},
            ("Mutant", "Kinematic"):  {"color": "tab:orange", "linestyle": "-"},
            ("Mutant", "Plant"):      {"color": "tab:orange", "linestyle": "--"},
        }
        fallback_colors = ["tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
        color_idx = 0

        xmin_g, xmax_g, ymin_g, ymax_g = None, None, None, None
        any_plotted = False
        for key in sorted(groups.keys()):
            plist = groups[key]
            result = self._reconstruct_group_curve(plist, metric, use_sem)
            if result is None:
                continue

            cond, scope = key
            style = style_map.get(key, None)
            if style is None:
                style = {"color": fallback_colors[color_idx % len(fallback_colors)], "linestyle": "-"}
                color_idx += 1

            label = f"{cond} \u2013 {scope} (n={result['n_roots']})"
            ax.plot(result["x_grid"], result["y_mean"],
                    color=style["color"], linestyle=style["linestyle"],
                    linewidth=2.0, label=label)
            ax.fill_between(result["x_grid"], result["y_minus"], result["y_plus"],
                            color=style["color"], alpha=0.12)

            for arr in [result["y_mean"], result["y_plus"], result["y_minus"]]:
                m = np.isfinite(arr)
                if np.any(m):
                    lo, hi = float(np.min(arr[m])), float(np.max(arr[m]))
                    ymin_g = lo if ymin_g is None else min(ymin_g, lo)
                    ymax_g = hi if ymax_g is None else max(ymax_g, hi)
            xg = result["x_grid"]
            xmin_g = float(xg[0]) if xmin_g is None else min(xmin_g, float(xg[0]))
            xmax_g = float(xg[-1]) if xmax_g is None else max(xmax_g, float(xg[-1]))
            any_plotted = True

        if not any_plotted:
            self._show_info("No fit data", "No logistic fit parameters available for the selected groups.")
            return

        ylabel = "Velocity (\u00b5m/min)" if metric == "Velocity" else "REGR (1/min)"
        title = f"Scope Comparison \u2014 {metric}"
        if time_sel != "All":
            title += f" @ {time_sel}"
        ax.set_xlabel("Position (\u00b5m)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        if xmin_g is not None and xmax_g is not None and ymin_g is not None and ymax_g is not None:
            xpad = (xmax_g - xmin_g) * 0.05 or 1.0
            ypad = (ymax_g - ymin_g) * 0.05 or 1.0
            ax.set_xlim(xmin_g - xpad, xmax_g + xpad)
            ax.set_ylim(ymin_g - ypad, ymax_g + ypad)

    def _plot_regr_over_time(self, ax, metric: str, use_sem: bool) -> None:
        cond_sel = self.combo_common_condition.currentText()

        groups: Dict[Tuple[str, str], List[Profile]] = {}
        for p in self.profiles:
            cond = p.condition or "Unknown"
            if cond_sel != "All" and cond != cond_sel:
                continue
            tlabel = self._format_time_label(p.time_min) if p.time_min is not None else "Unknown"
            groups.setdefault((cond, tlabel), []).append(p)

        if not groups:
            self._show_info("No data", "No profiles match the selected condition.")
            return

        cond_colors = {
            "Control": "Blues", "Drought": "Reds",
            "WT": "Blues", "Mutant": "Oranges",
            "Salt": "Greens", "Treated": "Purples",
        }

        all_conds = sorted({k[0] for k in groups.keys()})
        all_times_raw: set = set()
        for k in groups.keys():
            all_times_raw.add(k[1])

        def _time_sort_key(t: str) -> float:
            try:
                return float(t.replace(" min", ""))
            except ValueError:
                return float("inf")
        all_times = sorted(all_times_raw, key=_time_sort_key)

        xmin_g, xmax_g, ymin_g, ymax_g = None, None, None, None
        any_plotted = False

        for cond in all_conds:
            cmap_name = cond_colors.get(cond, "Greys")
            import matplotlib
            cmap = matplotlib.colormaps.get_cmap(cmap_name)

            cond_times = [t for t in all_times if (cond, t) in groups]
            n_times = len(cond_times)

            for ti, tlabel in enumerate(cond_times):
                plist = groups[(cond, tlabel)]
                result = self._reconstruct_group_curve(plist, metric, use_sem)
                if result is None:
                    continue

                if n_times > 1:
                    frac = 0.3 + 0.6 * ti / (n_times - 1)
                else:
                    frac = 0.6
                color = cmap(frac)

                label = f"{cond} @ {tlabel} (n={result['n_roots']})"
                ax.plot(result["x_grid"], result["y_mean"],
                        color=color, linewidth=2.0, label=label)
                ax.fill_between(result["x_grid"], result["y_minus"], result["y_plus"],
                                color=color, alpha=0.12)

                for arr in [result["y_mean"], result["y_plus"], result["y_minus"]]:
                    m = np.isfinite(arr)
                    if np.any(m):
                        lo, hi = float(np.min(arr[m])), float(np.max(arr[m]))
                        ymin_g = lo if ymin_g is None else min(ymin_g, lo)
                        ymax_g = hi if ymax_g is None else max(ymax_g, hi)
                xg = result["x_grid"]
                xmin_g = float(xg[0]) if xmin_g is None else min(xmin_g, float(xg[0]))
                xmax_g = float(xg[-1]) if xmax_g is None else max(xmax_g, float(xg[-1]))
                any_plotted = True

        if not any_plotted:
            self._show_info("No fit data", "No logistic fit parameters available for the selected groups.")
            return

        ylabel = "Velocity (\u00b5m/min)" if metric == "Velocity" else "REGR (1/min)"
        title = f"{metric} over Time"
        if cond_sel != "All":
            title += f" \u2014 {cond_sel}"
        ax.set_xlabel("Position (\u00b5m)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        if xmin_g is not None and xmax_g is not None and ymin_g is not None and ymax_g is not None:
            xpad = (xmax_g - xmin_g) * 0.05 or 1.0
            ypad = (ymax_g - ymin_g) * 0.05 or 1.0
            ax.set_xlim(xmin_g - xpad, xmax_g + xpad)
            ax.set_ylim(ymin_g - ypad, ymax_g + ypad)

    # ------------------------------------------------------------------
    # Statistical Analysis
    # ------------------------------------------------------------------
    def _extract_scalar_metric(self, p: "Profile", metric_name: str) -> Optional[float]:
        def _logistic3(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return a / (1.0 + np.exp(-b * (x - c)))

        if p.logistic_fit_3p is not None:
            params = p.logistic_fit_3p
            x_dom = p.regr_x if p.regr_x is not None else p.x_um
            if x_dom.size == 0:
                return None
            v_fit = _logistic3(x_dom, *params)
            regr_fit = np.gradient(v_fit, x_dom)
        elif p.logistic_params is not None:
            lp = p.logistic_params
            x_dom = p.x_um
            if x_dom.size == 0:
                return None
            v_fit = logistic_velocity(x_dom, lp)
            regr_fit = np.gradient(v_fit, x_dom)
        else:
            return None

        if metric_name == "Max Velocity":
            return float(np.nanmax(v_fit))
        elif metric_name == "Max REGR":
            return float(np.nanmax(regr_fit))
        elif metric_name == "REGR Peak Position":
            idx = int(np.nanargmax(regr_fit))
            return float(x_dom[idx])
        elif metric_name == "REGR AUC":
            return float(np.trapz(regr_fit, x_dom))
        return None

    @staticmethod
    def _significance_label(pval: float) -> str:
        if pval < 0.001:
            return "***"
        if pval < 0.01:
            return "**"
        if pval < 0.05:
            return "*"
        return "n.s."

    def _on_plot_statistics(self) -> None:
        self._pull_table_edits()
        if not self.profiles:
            self._show_info("No data", "No profiles loaded.")
            return

        plot_type = self.combo_stat_plot.currentText()
        if plot_type == "Trajectory":
            self._plot_trajectory()
            return

        from scipy import stats as sp_stats

        mode = self.combo_stat_mode.currentText()

        if mode == "Scope":
            self._plot_scope_grid()
            return
        metric = self.combo_stat_metric.currentText()
        test_name = self.combo_stat_test.currentText()
        time_sel = self.combo_stat_time.currentText()
        cond_sel = self.combo_stat_cond.currentText()
        scope_sel = self.combo_stat_scope.currentText()

        filtered: List[Profile] = []
        for p in self.profiles:
            if mode != "Time Point" and time_sel != "All":
                tlabel = self._format_time_label(p.time_min) if p.time_min is not None else "Unknown"
                if tlabel != time_sel:
                    continue
            if mode != "Condition" and cond_sel != "All":
                if (p.condition or "Unknown") != cond_sel:
                    continue
            if mode != "Scope" and scope_sel != "All":
                scope_val = (
                    p.scope_type.value
                    if p.scope_type and p.scope_type != ScopeType.AUTOMATIC
                    else "Unknown"
                )
                if scope_val != scope_sel:
                    continue
            filtered.append(p)

        if not filtered:
            self._show_info("No data", "No profiles match the selected filters.")
            return

        groups: Dict[str, List[float]] = {}
        for p in filtered:
            if mode == "Condition":
                key = p.condition or "Unknown"
            elif mode == "Scope":
                key = (
                    p.scope_type.value
                    if p.scope_type and p.scope_type != ScopeType.AUTOMATIC
                    else "Unknown"
                )
            else:
                key = self._format_time_label(p.time_min) if p.time_min is not None else "Unknown"

            val = self._extract_scalar_metric(p, metric)
            if val is not None:
                groups.setdefault(key, []).append(val)

        if not groups:
            self._show_info("No fit data", "No logistic fit data available for the selected profiles.")
            return

        if mode == "Time Point":
            def _tsort(t: str) -> float:
                try:
                    return float(t.replace(" min", ""))
                except ValueError:
                    return float("inf")
            sorted_keys = sorted(groups.keys(), key=_tsort)
        else:
            sorted_keys = sorted(groups.keys())

        group_arrays = [np.array(groups[k]) for k in sorted_keys]

        fig = self.canvas_stats.figure
        fig.clear()
        ax = fig.add_subplot(111)
        self.canvas_stats.ax = ax

        palette = [
            "#4C72B0", "#DD8452", "#55A868", "#C44E52",
            "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
        ]
        rng = np.random.default_rng(42)

        metric_units = {
            "Max Velocity": "\u00b5m/min",
            "Max REGR": "1/min",
            "REGR Peak Position": "\u00b5m",
            "REGR AUC": "\u00b5m\u207B\u00b9",
        }
        ylabel = f"{metric} ({metric_units.get(metric, '')})"

        if plot_type == "Bar + Points":
            x_pos = np.arange(len(sorted_keys))
            means = [np.mean(g) for g in group_arrays]
            sems = [
                np.std(g, ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0
                for g in group_arrays
            ]
            ax.bar(
                x_pos, means, yerr=sems, capsize=5,
                color=[palette[i % len(palette)] for i in range(len(sorted_keys))],
                alpha=0.7, edgecolor="black", linewidth=0.8, zorder=2,
            )
            for i, (k, arr) in enumerate(zip(sorted_keys, group_arrays)):
                jitter = rng.uniform(-0.15, 0.15, size=len(arr))
                ax.scatter(
                    np.full(len(arr), i) + jitter, arr,
                    color="black", s=18, zorder=5, alpha=0.6,
                )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sorted_keys)
        else:  # Box Plot
            bp = ax.boxplot(
                [groups[k] for k in sorted_keys],
                patch_artist=True, labels=sorted_keys, widths=0.55,
            )
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(palette[i % len(palette)])
                patch.set_alpha(0.7)
            for i, (k, arr) in enumerate(zip(sorted_keys, group_arrays)):
                jitter = rng.uniform(-0.08, 0.08, size=len(arr))
                ax.scatter(
                    np.full(len(arr), i + 1) + jitter, arr,
                    color="black", s=18, zorder=5, alpha=0.6,
                )

        ax.set_ylabel(ylabel)
        filter_parts = []
        if mode != "Time Point" and time_sel != "All":
            filter_parts.append(time_sel)
        if mode != "Condition" and cond_sel != "All":
            filter_parts.append(cond_sel)
        if mode != "Scope" and scope_sel != "All":
            filter_parts.append(scope_sel)
        subtitle = " | ".join(filter_parts) if filter_parts else "all data"
        ax.set_title(f"{metric} \u2014 by {mode}\n({subtitle})", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        results_lines: List[str] = []
        results_lines.append(f"=== {metric} \u2014 Compare by {mode} ===")
        results_lines.append(f"Filters: time={time_sel}, condition={cond_sel}, scope={scope_sel}")
        results_lines.append("")

        for k, arr in zip(sorted_keys, group_arrays):
            sd = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
            results_lines.append(
                f"  {k:20s}  n={len(arr):3d}  mean={np.mean(arr):.4f}  SD={sd:.4f}"
            )
        results_lines.append("")

        pval_main: Optional[float] = None
        stat_main: Optional[float] = None

        if len(group_arrays) >= 2:
            min_n = min(len(g) for g in group_arrays)

            if len(group_arrays) == 2:
                a, b = group_arrays
                if test_name == "t-test (Welch)":
                    if min_n >= 2:
                        stat_main, pval_main = sp_stats.ttest_ind(a, b, equal_var=False)
                        results_lines.append(
                            f"Welch's t-test: t = {stat_main:.4f}, p = {pval_main:.4e}  "
                            f"{self._significance_label(pval_main)}"
                        )
                    else:
                        results_lines.append("t-test: insufficient data (n < 2 in a group)")
                elif test_name == "Mann-Whitney U":
                    if min_n >= 1:
                        stat_main, pval_main = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
                        results_lines.append(
                            f"Mann-Whitney U: U = {stat_main:.4f}, p = {pval_main:.4e}  "
                            f"{self._significance_label(pval_main)}"
                        )
                    else:
                        results_lines.append("Mann-Whitney U: insufficient data")
                elif test_name == "One-way ANOVA":
                    if min_n >= 2:
                        stat_main, pval_main = sp_stats.f_oneway(*group_arrays)
                        results_lines.append(
                            f"One-way ANOVA: F = {stat_main:.4f}, p = {pval_main:.4e}  "
                            f"{self._significance_label(pval_main)}"
                        )
                    else:
                        results_lines.append("ANOVA: insufficient data (n < 2 in a group)")
                elif test_name == "Kruskal-Wallis":
                    if min_n >= 1:
                        stat_main, pval_main = sp_stats.kruskal(*group_arrays)
                        results_lines.append(
                            f"Kruskal-Wallis: H = {stat_main:.4f}, p = {pval_main:.4e}  "
                            f"{self._significance_label(pval_main)}"
                        )
                    else:
                        results_lines.append("Kruskal-Wallis: insufficient data")

            else:  # 3+ groups
                if test_name in ("t-test (Welch)", "One-way ANOVA"):
                    if min_n >= 2:
                        stat_main, pval_main = sp_stats.f_oneway(*group_arrays)
                        results_lines.append(
                            f"One-way ANOVA: F = {stat_main:.4f}, p = {pval_main:.4e}  "
                            f"{self._significance_label(pval_main)}"
                        )
                    else:
                        results_lines.append("ANOVA: insufficient data (n < 2 in a group)")
                elif test_name in ("Mann-Whitney U", "Kruskal-Wallis"):
                    if min_n >= 1:
                        stat_main, pval_main = sp_stats.kruskal(*group_arrays)
                        results_lines.append(
                            f"Kruskal-Wallis: H = {stat_main:.4f}, p = {pval_main:.4e}  "
                            f"{self._significance_label(pval_main)}"
                        )
                    else:
                        results_lines.append("Kruskal-Wallis: insufficient data")

                n_comp = len(sorted_keys) * (len(sorted_keys) - 1) // 2
                if n_comp > 0 and min_n >= 2:
                    results_lines.append("")
                    results_lines.append("Pairwise post-hoc (Bonferroni corrected):")
                    for i in range(len(sorted_keys)):
                        for j in range(i + 1, len(sorted_keys)):
                            gi, gj = group_arrays[i], group_arrays[j]
                            if test_name in ("t-test (Welch)", "One-way ANOVA"):
                                ts, pv = sp_stats.ttest_ind(gi, gj, equal_var=False)
                            else:
                                ts, pv = sp_stats.mannwhitneyu(gi, gj, alternative="two-sided")
                            pv_corr = min(pv * n_comp, 1.0)
                            results_lines.append(
                                f"  {sorted_keys[i]} vs {sorted_keys[j]}: "
                                f"p = {pv_corr:.4e}  {self._significance_label(pv_corr)}"
                            )

        if pval_main is not None and len(group_arrays) == 2:
            all_vals = np.concatenate(group_arrays)
            y_top = float(np.max(all_vals))
            y_range = float(np.ptp(all_vals)) or 1.0
            bracket_y = y_top + y_range * 0.08

            if plot_type == "Bar + Points":
                x1, x2 = 0, 1
            else:
                x1, x2 = 1, 2

            ax.plot(
                [x1, x1, x2, x2],
                [bracket_y, bracket_y + y_range * 0.02, bracket_y + y_range * 0.02, bracket_y],
                lw=1.2, color="black",
            )
            sig_text = f"p = {pval_main:.3e} {self._significance_label(pval_main)}"
            ax.text(
                (x1 + x2) / 2, bracket_y + y_range * 0.03, sig_text,
                ha="center", va="bottom", fontsize=9,
            )

        ax.figure.tight_layout()
        self.canvas_stats.draw()

        self.stats_results.setPlainText("\n".join(results_lines))
        self.log(f"Statistical analysis complete: {metric} by {mode}.")

    def _plot_scope_grid(self) -> None:
        from scipy import stats as sp_stats

        metric = self.combo_stat_metric.currentText()
        test_name = self.combo_stat_test.currentText()

        metric_units = {
            "Max Velocity": "\u00b5m/min", "Max REGR": "1/min",
            "REGR Peak Position": "\u00b5m", "REGR AUC": "\u00b5m\u207B\u00b9",
        }
        ylabel = f"{metric} ({metric_units.get(metric, '')})"

        groups: Dict[Tuple[str, str], List[float]] = {}
        for p in self.profiles:
            cond = p.condition or "Unknown"
            scope_val = (
                p.scope_type.value
                if p.scope_type and p.scope_type != ScopeType.AUTOMATIC
                else "Unknown"
            )
            val = self._extract_scalar_metric(p, metric)
            if val is None:
                continue
            groups.setdefault((cond, scope_val), []).append(val)

        if not groups:
            self._show_info("No fit data", "No logistic fit data available.")
            return

        sorted_combos = sorted(groups.keys())

        cond_palette = {
            "Control": "#4C72B0", "WT": "#4C72B0",
            "Drought": "#C44E52", "Mutant": "#DD8452",
            "Salt": "#55A868", "Treated": "#8172B3", "Perturbed": "#937860",
        }
        fb = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
        scope_hatches = {"Kinematic": "", "Plant": "///", "Unknown": "xx"}
        rng = np.random.default_rng(42)

        fig = self.canvas_stats.figure
        fig.clear()
        ax = fig.add_subplot(111)
        self.canvas_stats.ax = ax

        box_data = [groups[c] for c in sorted_combos]
        labels = [f"{c[0]}\n{c[1]}" for c in sorted_combos]
        colors = [cond_palette.get(c[0], fb[i % len(fb)]) for i, c in enumerate(sorted_combos)]
        hatches = [scope_hatches.get(c[1], "") for c in sorted_combos]

        bp = ax.boxplot(box_data, patch_artist=True, labels=labels, widths=0.55)
        for i, (patch, h, col) in enumerate(zip(bp["boxes"], hatches, colors)):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
            patch.set_hatch(h)

        for i, combo in enumerate(sorted_combos):
            arr = np.array(groups[combo])
            jitter = rng.uniform(-0.1, 0.1, size=len(arr))
            ax.scatter(
                np.full(len(arr), i + 1) + jitter, arr,
                color="black", s=16, zorder=5, alpha=0.5,
            )

        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} \u2014 Scope Comparison", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        results_lines: List[str] = []
        results_lines.append(f"=== {metric} \u2014 Scope Comparison ===")
        results_lines.append("")

        for combo in sorted_combos:
            cond, scope = combo
            vals = groups[combo]
            n = len(vals)
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            results_lines.append(
                f"  {cond:12s} {scope:10s}  n={n:3d}  mean={m:.4f}  SD={sd:.4f}"
            )
        results_lines.append("")

        combo_list = [(c, np.array(groups[c])) for c in sorted_combos if len(groups[c]) >= 2]
        if len(combo_list) >= 2:
            results_lines.append("Pairwise comparisons (Bonferroni corrected):")
            n_comp = len(combo_list) * (len(combo_list) - 1) // 2
            for i in range(len(combo_list)):
                for j in range(i + 1, len(combo_list)):
                    (c1, a), (c2, b) = combo_list[i], combo_list[j]
                    l1 = f"{c1[0]}-{c1[1]}"
                    l2 = f"{c2[0]}-{c2[1]}"
                    if test_name in ("t-test (Welch)", "One-way ANOVA"):
                        _, pv = sp_stats.ttest_ind(a, b, equal_var=False)
                    else:
                        _, pv = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
                    pv_corr = min(pv * n_comp, 1.0)
                    results_lines.append(
                        f"  {l1} vs {l2}: p={pv_corr:.4e} {self._significance_label(pv_corr)}"
                    )

        fig.tight_layout()
        self.canvas_stats.draw()
        self.stats_results.setPlainText("\n".join(results_lines))
        self.log(f"Scope comparison complete: {metric}, {len(sorted_combos)} groups.")

    def _plot_trajectory(self) -> None:
        from scipy import stats as sp_stats

        metric = self.combo_stat_metric.currentText()
        test_name = self.combo_stat_test.currentText()
        scope_sel = self.combo_stat_scope.currentText()

        filtered: List[Profile] = []
        for p in self.profiles:
            if scope_sel != "All":
                scope_val = (
                    p.scope_type.value
                    if p.scope_type and p.scope_type != ScopeType.AUTOMATIC
                    else "Unknown"
                )
                if scope_val != scope_sel:
                    continue
            filtered.append(p)

        if not filtered:
            self._show_info("No data", "No profiles match the selected scope filter.")
            return

        data: Dict[str, Dict[float, List[float]]] = {}
        for p in filtered:
            cond = p.condition or "Unknown"
            t = p.time_min
            if t is None:
                continue
            val = self._extract_scalar_metric(p, metric)
            if val is None:
                continue
            data.setdefault(cond, {}).setdefault(t, []).append(val)

        if not data:
            self._show_info("No fit data", "No profiles with time info and logistic fits available.")
            return

        all_raw_times: set[float] = set()
        for cond_dict in data.values():
            all_raw_times.update(cond_dict.keys())
        sorted_raw = sorted(all_raw_times)

        if not sorted_raw:
            self._show_info("No time points", "Need at least one time point.")
            return

        display_hours = {t: self._raw_time_to_hours(t) for t in sorted_raw}

        fig = self.canvas_stats.figure
        fig.clear()
        ax = fig.add_subplot(111)
        self.canvas_stats.ax = ax

        palette = {
            "Control": "#4C72B0", "WT": "#4C72B0",
            "Drought": "#C44E52", "Mutant": "#DD8452",
            "Salt": "#55A868", "Treated": "#8172B3", "Perturbed": "#937860",
        }
        fallback_colors = [
            "#4C72B0", "#DD8452", "#55A868", "#C44E52",
            "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
        ]
        rng = np.random.default_rng(42)

        metric_units = {
            "Max Velocity": "\u00b5m/min", "Max REGR": "1/min",
            "REGR Peak Position": "\u00b5m", "REGR AUC": "\u00b5m\u207B\u00b9",
        }
        ylabel = f"{metric} ({metric_units.get(metric, '')})"

        sorted_conds = sorted(data.keys())
        results_lines: List[str] = []
        results_lines.append(f"=== {metric} \u2014 Trajectory over Time ===")
        results_lines.append(f"Scope filter: {scope_sel}")
        results_lines.append("")

        hrs_vals = [display_hours[t] for t in sorted_raw]
        if len(hrs_vals) > 1:
            jitter_w = (hrs_vals[-1] - hrs_vals[0]) * 0.015
        else:
            jitter_w = 0.05

        for ci, cond in enumerate(sorted_conds):
            cond_data = data[cond]
            color = palette.get(cond, fallback_colors[ci % len(fallback_colors)])

            h_vals: List[float] = []
            means: List[float] = []
            sems: List[float] = []
            ns: List[int] = []

            for raw_t in sorted_raw:
                if raw_t not in cond_data:
                    continue
                arr = np.array(cond_data[raw_t])
                h = display_hours[raw_t]
                h_vals.append(h)
                means.append(float(np.mean(arr)))
                ns.append(len(arr))
                sems.append(
                    float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
                    if len(arr) > 1 else 0.0
                )

                jitter = rng.uniform(-jitter_w, jitter_w, size=len(arr))
                ax.scatter(
                    np.full(len(arr), h) + jitter, arr,
                    color=color, s=22, alpha=0.45, zorder=4, edgecolors="none",
                )

            if not h_vals:
                continue

            h_arr = np.array(h_vals)
            m_arr = np.array(means)
            sem_arr = np.array(sems)

            ax.plot(
                h_arr, m_arr, color=color, linewidth=2.2, marker="o",
                markersize=7, markeredgecolor="white", markeredgewidth=1.0,
                zorder=6, label=cond,
            )
            ax.errorbar(
                h_arr, m_arr, yerr=sem_arr, color=color, fmt="none",
                capsize=4, capthick=1.5, linewidth=1.5, zorder=5,
            )
            ax.fill_between(
                h_arr, m_arr - sem_arr, m_arr + sem_arr,
                color=color, alpha=0.10, zorder=2,
            )

            results_lines.append(f"  {cond}:")
            for h, m, s, n in zip(h_vals, means, sems, ns):
                results_lines.append(
                    f"    {h:g} hrs  n={n:3d}  mean={m:.4f}  SEM={s:.4f}"
                )
            results_lines.append("")

        if len(sorted_conds) >= 2:
            results_lines.append("--- Pairwise comparisons at each time point ---")
            for raw_t in sorted_raw:
                h = display_hours[raw_t]
                conds_at_t = [c for c in sorted_conds if raw_t in data[c]]
                if len(conds_at_t) < 2:
                    continue
                results_lines.append(f"  t = {h:g} hrs:")
                for i in range(len(conds_at_t)):
                    for j in range(i + 1, len(conds_at_t)):
                        c1, c2 = conds_at_t[i], conds_at_t[j]
                        a = np.array(data[c1][raw_t])
                        b = np.array(data[c2][raw_t])
                        if len(a) < 2 or len(b) < 2:
                            results_lines.append(f"    {c1} vs {c2}: insufficient data")
                            continue
                        if test_name in ("t-test (Welch)", "One-way ANOVA"):
                            ts, pv = sp_stats.ttest_ind(a, b, equal_var=False)
                            results_lines.append(
                                f"    {c1} vs {c2}: t={ts:.3f}, p={pv:.4e}  "
                                f"{self._significance_label(pv)}"
                            )
                        else:
                            ts, pv = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
                            results_lines.append(
                                f"    {c1} vs {c2}: U={ts:.1f}, p={pv:.4e}  "
                                f"{self._significance_label(pv)}"
                            )

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(ylabel)
        scope_note = f" ({scope_sel} scope)" if scope_sel != "All" else ""
        ax.set_title(f"{metric} \u2014 Trajectory over Time{scope_note}", fontsize=11)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        if hrs_vals:
            ax.set_xticks(hrs_vals)
            ax.set_xticklabels([f"{h:g}" for h in hrs_vals])

        fig.tight_layout()
        self.canvas_stats.draw()

        self.stats_results.setPlainText("\n".join(results_lines))
        self.log(f"Trajectory analysis complete: {metric} over time.")

    # ------------------------------------------------------------------
    # Exporting
    # ------------------------------------------------------------------
    def save_current_figure(self, fmt: str, model: bool = False, canvas_override=None) -> None:
        canvas = canvas_override or (self.canvas_model if model else self.canvas_raw)
        suffix = fmt.lower()
        if suffix not in {"png", "svg"}:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {suffix.upper()}",
            f"plot.{suffix}",
            f"{suffix.upper()} files (*.{suffix});;All files (*)",
        )
        if not path:
            return
        try:
            canvas.figure.savefig(path, format=suffix)
        except Exception as exc:
            self._show_error("Save failed", str(exc))

    def _summarize_group(self, plist: List[Profile], n_grid: int = 500) -> Optional[Dict[str, object]]:
        def _logistic3(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return a / (1.0 + np.exp(-b * (x - c)))

        fit_params: List[np.ndarray] = []
        curve_data: List[Dict[str, np.ndarray]] = []
        for p in plist:
            if p.logistic_fit_3p is not None:
                fit_params.append(p.logistic_fit_3p)
                x_dom = p.regr_x if p.regr_x is not None else p.x_um
                v_fit = _logistic3(x_dom, *p.logistic_fit_3p)
                regr_fit = np.gradient(v_fit, x_dom)
                curve_data.append({"lDomain": x_dom, "vel": v_fit, "regr": regr_fit})
            elif p.logistic_params is not None:
                lp = p.logistic_params
                pv = np.array([lp.v0, lp.L, lp.k, lp.x0])
                fit_params.append(pv)

        if not fit_params:
            return None

        P = np.array(fit_params, dtype=float)
        param_mean = np.nanmean(P, axis=0)
        param_sd = np.nanstd(P, axis=0, ddof=1) if P.shape[0] > 1 else np.zeros(P.shape[1])

        if curve_data:
            xmin = max(float(np.min(cd["lDomain"])) for cd in curve_data)
            xmax = min(float(np.max(cd["lDomain"])) for cd in curve_data)
        else:
            xmin = 0.0
            xmax = max(float(np.nanmax(p.x_um)) for p in plist if p.x_um.size) or 1000.0
        if xmax <= xmin:
            xmax = xmin + 1.0
        xgrid = np.linspace(xmin, xmax, n_grid)

        if P.shape[1] == 3:
            v_mean_p = _logistic3(xgrid, *param_mean)
            v_plus_p = _logistic3(xgrid, *(param_mean + param_sd))
            v_minus_p = _logistic3(xgrid, *(param_mean - param_sd))
        else:
            from .ki_models import params_from_array
            v_mean_p = logistic_velocity(xgrid, params_from_array(param_mean))
            v_plus_p = logistic_velocity(xgrid, params_from_array(param_mean + param_sd))
            v_minus_p = logistic_velocity(xgrid, params_from_array(param_mean - param_sd))
        regr_mean_p = np.gradient(v_mean_p, xgrid)
        regr_plus_p = np.gradient(v_plus_p, xgrid)
        regr_minus_p = np.gradient(v_minus_p, xgrid)

        if curve_data:
            V_all = np.vstack([np.interp(xgrid, cd["lDomain"], cd["vel"]) for cd in curve_data])
            S_all = np.vstack([np.interp(xgrid, cd["lDomain"], cd["regr"]) for cd in curve_data])
            v_mean_pw = np.nanmean(V_all, axis=0)
            v_sd_pw = np.nanstd(V_all, axis=0, ddof=1) if V_all.shape[0] > 1 else np.zeros_like(v_mean_pw)
            regr_mean_pw = np.nanmean(S_all, axis=0)
            regr_sd_pw = np.nanstd(S_all, axis=0, ddof=1) if S_all.shape[0] > 1 else np.zeros_like(regr_mean_pw)
        else:
            v_mean_pw = v_sd_pw = regr_mean_pw = regr_sd_pw = np.zeros(n_grid)

        v_max = float(np.nanmax(v_mean_p))
        regr_max = float(np.nanmax(regr_mean_p))
        regr_max_x = float(xgrid[int(np.nanargmax(regr_mean_p))])
        regr_auc = float(np.trapz(regr_mean_p, xgrid))

        return {
            "n": P.shape[0],
            "param_mean": param_mean,
            "param_sd": param_sd,
            "xgrid": xgrid,
            "v_mean_param": v_mean_p,
            "v_plus_param": v_plus_p,
            "v_minus_param": v_minus_p,
            "regr_mean_param": regr_mean_p,
            "regr_plus_param": regr_plus_p,
            "regr_minus_param": regr_minus_p,
            "v_mean_pw": v_mean_pw,
            "v_sd_pw": v_sd_pw,
            "regr_mean_pw": regr_mean_pw,
            "regr_sd_pw": regr_sd_pw,
            "v_max": v_max,
            "regr_max": regr_max,
            "regr_max_x": regr_max_x,
            "regr_auc": regr_auc,
        }

    def on_export_summary(self) -> None:
        if not self.profiles:
            self._show_info("No data", "No profiles loaded.")
            return

        groups: Dict[Tuple, List[Profile]] = {}
        for p in self.profiles:
            if self.mode == "genetic":
                key = group_key_genetic(p)
            else:
                key = group_key_environmental(p)
            groups.setdefault(key, []).append(p)

        summaries: Dict[str, Dict[str, object]] = {}
        for key, plist in sorted(groups.items()):
            label = " / ".join(str(k) for k in key)
            s = self._summarize_group(plist)
            if s is not None:
                summaries[label] = s

        if not summaries:
            self._show_info("No fit data", "No logistic fit parameters found. Nothing to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save summary CSV (curves CSV saved alongside)",
            "parameters_summary.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        import csv
        from pathlib import Path as _Path

        base = _Path(path)
        params_path = base
        curves_path = base.parent / (base.stem + "_curves" + base.suffix)

        param_rows: List[Dict[str, object]] = []
        for label, s in summaries.items():
            pm = s["param_mean"]
            ps = s["param_sd"]
            n_params = len(pm)
            row: Dict[str, object] = {"group": label, "n_profiles": s["n"]}
            if n_params == 3:
                row.update({
                    "a_mean": pm[0], "b_mean": pm[1], "c_mean": pm[2],
                    "a_sd": ps[0], "b_sd": ps[1], "c_sd": ps[2],
                })
            else:
                row.update({
                    "v0_mean": pm[0], "L_mean": pm[1], "k_mean": pm[2], "x0_mean": pm[3],
                    "v0_sd": ps[0], "L_sd": ps[1], "k_sd": ps[2], "x0_sd": ps[3],
                })
            row.update({
                "v_max": s["v_max"],
                "regr_max": s["regr_max"],
                "regr_max_x_um": s["regr_max_x"],
                "regr_auc": s["regr_auc"],
            })
            param_rows.append(row)

        try:
            fieldnames = list(param_rows[0].keys())
            with open(params_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in param_rows:
                    writer.writerow(row)
        except Exception as exc:
            self._show_error("Export failed (params)", str(exc))
            return

        try:
            first_key = next(iter(summaries))
            xgrid = summaries[first_key]["xgrid"]
            curve_cols: Dict[str, np.ndarray] = {"x_um": xgrid}
            for label, s in summaries.items():
                prefix = label.replace(" ", "_").replace("/", "_")
                for key in ["v_mean_param", "v_plus_param", "v_minus_param",
                            "regr_mean_param", "regr_plus_param", "regr_minus_param",
                            "v_mean_pw", "v_sd_pw", "regr_mean_pw", "regr_sd_pw"]:
                    curve_cols[f"{prefix}_{key}"] = s[key]

            col_names = list(curve_cols.keys())
            n_rows = len(xgrid)
            with open(curves_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(col_names)
                for i in range(n_rows):
                    writer.writerow([float(curve_cols[c][i]) for c in col_names])
        except Exception as exc:
            self._show_error("Export failed (curves)", str(exc))
            return

        self.log(
            f"Exported parameter summary ({len(param_rows)} groups) to {params_path}\n"
            f"Exported curve data ({n_rows} points) to {curves_path}"
        )

    # ------------------------------------------------------------------
    # Parameter mapping dialog
    # ------------------------------------------------------------------
    def on_edit_mapping(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Logistic parameter mapping")
        layout = QGridLayout(dlg)

        layout.addWidget(QLabel("CSV column for v0:"), 0, 0)
        edit_v0 = QLineEdit(self.logistic_mapping.v0 or "")
        layout.addWidget(edit_v0, 0, 1)

        layout.addWidget(QLabel("CSV column for L:"), 1, 0)
        edit_L = QLineEdit(self.logistic_mapping.L or "")
        layout.addWidget(edit_L, 1, 1)

        layout.addWidget(QLabel("CSV column for k:"), 2, 0)
        edit_k = QLineEdit(self.logistic_mapping.k or "")
        layout.addWidget(edit_k, 2, 1)

        layout.addWidget(QLabel("CSV column for x0:"), 3, 0)
        edit_x0 = QLineEdit(self.logistic_mapping.x0 or "")
        layout.addWidget(edit_x0, 3, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons, 4, 0, 1, 2)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() == QDialog.Accepted:
            self.logistic_mapping = LogisticMapping(
                v0=edit_v0.text().strip() or None,
                L=edit_L.text().strip() or None,
                k=edit_k.text().strip() or None,
                x0=edit_x0.text().strip() or None,
            )
            self._save_mapping_config()
            self.log("Updated logistic parameter mapping (used on next reload).")

    def _mapping_config_path(self) -> Path:
        return Path.cwd() / "logistic_mapping.json"

    def _save_mapping_config(self) -> None:
        path = self._mapping_config_path()
        data = self.logistic_mapping.to_dict()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_mapping_config(self) -> None:
        path = self._mapping_config_path()
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logistic_mapping = LogisticMapping.from_dict(data)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Logging & errors
    # ------------------------------------------------------------------
    def log(self, text: str) -> None:
        self.log_widget.append(text)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.log(f"ERROR: {title}: {message}")

    def _show_info(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)
        self.log(f"{title}: {message}")
