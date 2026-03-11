"""Modern PySide6 UI for the integrated pipeline."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .analysis_widget import AnalysisWidget
from .auto_midline import AutoMidlineResult
from .data_loader import discover_output_profiles
from .manual_input import trace_midline_on_image
from .pipeline_runner import PipelineResult, run_batch


class PipelineWorker(QObject):
    finished = Signal(list)
    failed = Signal(str)
    status = Signal(str)
    request_auto_review = Signal(object, object)
    request_manual_trace = Signal(object, float)

    def __init__(
        self,
        root_folder: str,
        output_root: str,
        mode: str,
        checkpoint_path: str | None,
        spacing_px: float,
        disk_radius: int,
        threshold: float,
        time_interval: float,
        time_unit: str,
    ):
        super().__init__()
        self.root_folder = root_folder
        self.output_root = output_root
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.spacing_px = spacing_px
        self.disk_radius = disk_radius
        self.threshold = threshold
        self.time_interval = time_interval
        self.time_unit = time_unit

        self._auto_event = threading.Event()
        self._manual_event = threading.Event()
        self._auto_response = False
        self._manual_response: np.ndarray | None = None

    @Slot()
    def run(self) -> None:
        try:
            results = run_batch(
                root_folder=self.root_folder,
                output_root=self.output_root,
                mode=self.mode,
                checkpoint_path=self.checkpoint_path,
                spacing_px=self.spacing_px,
                disk_radius=self.disk_radius,
                threshold=self.threshold,
                time_interval=self.time_interval,
                time_unit=self.time_unit,
                auto_review_callback=self._auto_review_callback,
                manual_trace_callback=self._manual_trace_callback,
                progress_callback=lambda text: self.status.emit(text),
            )
            self.finished.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _auto_review_callback(self, first_frame: np.ndarray, auto_result: AutoMidlineResult) -> bool:
        self._auto_event.clear()
        self.request_auto_review.emit(first_frame, auto_result)
        self._auto_event.wait()
        return self._auto_response

    def _manual_trace_callback(self, first_frame: np.ndarray, spacing_px: float) -> np.ndarray:
        self._manual_event.clear()
        self.request_manual_trace.emit(first_frame, spacing_px)
        self._manual_event.wait()
        if self._manual_response is None:
            raise ValueError("Manual tracing was cancelled.")
        return self._manual_response

    def set_auto_response(self, approved: bool) -> None:
        self._auto_response = approved
        self._auto_event.set()

    def set_manual_response(self, points_xy: np.ndarray | None) -> None:
        self._manual_response = points_xy
        self._manual_event.set()


class AutoReviewDialog(QDialog):
    def __init__(self, overlay: np.ndarray, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 600)
        layout = QVBoxLayout(self)

        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        pix = QPixmap.fromImage(_np_to_qimage(overlay))
        label.setPixmap(pix.scaled(860, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(label)

        btn_row = QHBoxLayout()
        self.approve_btn = QPushButton("Approve")
        self.deny_btn = QPushButton("Deny")
        self.approve_btn.clicked.connect(self.accept)
        self.deny_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.approve_btn)
        btn_row.addWidget(self.deny_btn)
        layout.addLayout(btn_row)


class _TraceLabel(QLabel):
    """QLabel that collects click-traced points as image-space (x, y) coordinates."""

    def __init__(self, base_pixmap: QPixmap, scale: float, parent=None):
        super().__init__(parent)
        self._base_pixmap = base_pixmap
        self._scale = scale
        self._points: list[tuple[float, float]] = []
        self.setPixmap(base_pixmap)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            lx = event.position().x()
            ly = event.position().y()
            ix = lx / self._scale
            iy = ly / self._scale
            self._points.append((ix, iy))
            self._redraw()
        elif event.button() == Qt.RightButton and len(self._points) >= 2:
            p = self.parent()
            while p is not None:
                if isinstance(p, ManualTraceDialog):
                    p.accept()
                    return
                p = p.parent()

    def _redraw(self) -> None:
        canvas = self._base_pixmap.copy()
        painter = QPainter(canvas)
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        for i in range(1, len(self._points)):
            x0 = int(self._points[i - 1][0] * self._scale)
            y0 = int(self._points[i - 1][1] * self._scale)
            x1 = int(self._points[i][0] * self._scale)
            y1 = int(self._points[i][1] * self._scale)
            painter.drawLine(x0, y0, x1, y1)
        dot_pen = QPen(QColor(255, 0, 0), 6)
        painter.setPen(dot_pen)
        for px, py in self._points:
            painter.drawPoint(int(px * self._scale), int(py * self._scale))
        painter.end()
        self.setPixmap(canvas)

    def clear_points(self) -> None:
        self._points = []
        self.setPixmap(self._base_pixmap)

    def points(self) -> list[tuple[float, float]]:
        return list(self._points)


class ManualTraceDialog(QDialog):
    """Dialog for click-tracing a midline on the first frame image."""

    _MAX_W = 860
    _MAX_H = 520

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual midline trace")
        self.setModal(True)

        qimg = _np_to_qimage(image)
        full_pix = QPixmap.fromImage(qimg)
        scaled_pix = full_pix.scaled(self._MAX_W, self._MAX_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        iw = full_pix.width() or 1
        ih = full_pix.height() or 1
        sw = scaled_pix.width()
        sh = scaled_pix.height()
        scale = min(sw / iw, sh / ih)

        layout = QVBoxLayout(self)
        instructions = QLabel(
            "Left-click to trace the root midline (tip \u2192 base).\n"
            "Right-click or press Done when finished (\u22652 points required)."
        )
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)

        self._trace_label = _TraceLabel(base_pixmap=scaled_pix, scale=scale, parent=self)
        self._trace_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._trace_label)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        done_btn = QPushButton("Done")
        clear_btn.clicked.connect(self._on_clear)
        done_btn.clicked.connect(self._on_done)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(done_btn)
        layout.addLayout(btn_row)

    def _on_clear(self) -> None:
        self._trace_label.clear_points()

    def _on_done(self) -> None:
        if len(self._trace_label.points()) < 2:
            QMessageBox.warning(self, "Not enough points", "Click at least 2 points along the root midline.")
            return
        self.accept()

    def points_image_xy(self) -> list[tuple[float, float]]:
        return self._trace_label.points()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("All-in-One Root Kinematics")
        self.resize(1280, 820)
        self._worker: PipelineWorker | None = None
        self._worker_thread: QThread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        root_layout.addWidget(self._tabs)

        # --- Pipeline tab ---
        pipeline_widget = QWidget()
        pipeline_layout = QVBoxLayout(pipeline_widget)

        form_wrap = QWidget()
        form = QFormLayout(form_wrap)
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit(str(Path.cwd() / "integrated_outputs"))
        self.ckpt_edit = QLineEdit()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["auto", "manual"])
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(1.0, 500.0)
        self.spacing_spin.setValue(15.0)
        self.spacing_spin.setDecimals(1)
        self.disk_spin = QSpinBox()
        self.disk_spin.setRange(1, 500)
        self.disk_spin.setValue(28)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 500.0)
        self.threshold_spin.setValue(10.0)
        self.time_interval_spin = QDoubleSpinBox()
        self.time_interval_spin.setRange(0.001, 1_000_000.0)
        self.time_interval_spin.setValue(1.0)
        self.time_unit_edit = QLineEdit("frame")

        form.addRow("TIFF root folder", _with_browse_button(self.input_edit, self._pick_input_dir))
        form.addRow("Output folder", _with_browse_button(self.output_edit, self._pick_output_dir))
        form.addRow("Auto checkpoint (optional)", _with_browse_button(self.ckpt_edit, self._pick_checkpoint_file))

        # Checkpoint status label (below the checkpoint row)
        self._ckpt_status = QLabel("No checkpoint \u2014 classical auto-detection will be used")
        self._ckpt_status.setStyleSheet("color: gray;")
        form.addRow("", self._ckpt_status)
        self.ckpt_edit.textChanged.connect(self._update_ckpt_status)

        form.addRow("Initialization mode", self.mode_combo)
        form.addRow("Point spacing (px)", self.spacing_spin)
        form.addRow("Disk radius", self.disk_spin)
        form.addRow("Threshold", self.threshold_spin)
        form.addRow("Time interval", self.time_interval_spin)
        form.addRow("Time unit", self.time_unit_edit)
        pipeline_layout.addWidget(form_wrap)

        actions = QHBoxLayout()
        self.run_btn = QPushButton("Run integrated pipeline")
        self.refresh_btn = QPushButton("Refresh summary")
        self.status_label = QLabel("Ready.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.run_btn.clicked.connect(self._start_run)
        self.refresh_btn.clicked.connect(self._refresh_summary)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.refresh_btn)
        actions.addWidget(self.status_label, stretch=1)
        actions.addWidget(self.progress)
        pipeline_layout.addLayout(actions)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("Run the pipeline to populate results.")
        pipeline_layout.addWidget(self.summary, stretch=1)

        self._tabs.addTab(pipeline_widget, "\U0001F52C Pipeline")

        # --- Analysis tab ---
        self._analysis_widget = AnalysisWidget()
        self._tabs.addTab(self._analysis_widget, "\U0001F4CA Analysis")

    @Slot(str)
    def _update_ckpt_status(self, text: str) -> None:
        path = text.strip()
        if not path:
            self._ckpt_status.setText("No checkpoint \u2014 classical auto-detection will be used")
            self._ckpt_status.setStyleSheet("color: gray;")
        elif not Path(path).exists():
            self._ckpt_status.setText("\u26a0 File not found \u2014 classical fallback will be used")
            self._ckpt_status.setStyleSheet("color: orange;")
        else:
            self._ckpt_status.setText("\u2713 Checkpoint found")
            self._ckpt_status.setStyleSheet("color: green;")

    def _pick_input_dir(self) -> None:
        value = QFileDialog.getExistingDirectory(self, "Select TIFF root folder")
        if value:
            self.input_edit.setText(value)

    def _pick_output_dir(self) -> None:
        value = QFileDialog.getExistingDirectory(self, "Select output folder")
        if value:
            self.output_edit.setText(value)

    def _pick_checkpoint_file(self) -> None:
        value, _ = QFileDialog.getOpenFileName(self, "Select model checkpoint", filter="PyTorch (*.pth);;All files (*.*)")
        if value:
            self.ckpt_edit.setText(value)

    def _start_run(self) -> None:
        root_folder = self.input_edit.text().strip()
        if not root_folder:
            QMessageBox.warning(self, "Missing input", "Select a TIFF root folder first.")
            return
        Path(self.output_edit.text()).mkdir(parents=True, exist_ok=True)
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_label.setText("Running...")

        self._worker_thread = QThread()
        self._worker = PipelineWorker(
            root_folder=root_folder,
            output_root=self.output_edit.text().strip(),
            mode=self.mode_combo.currentText().strip().lower(),
            checkpoint_path=self.ckpt_edit.text().strip() or None,
            spacing_px=float(self.spacing_spin.value()),
            disk_radius=int(self.disk_spin.value()),
            threshold=float(self.threshold_spin.value()),
            time_interval=float(self.time_interval_spin.value()),
            time_unit=self.time_unit_edit.text().strip() or "frame",
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.status.connect(self._on_status)
        self._worker.request_auto_review.connect(self._on_request_auto_review)
        self._worker.request_manual_trace.connect(self._on_request_manual_trace)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()

    @Slot(str)
    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)

    @Slot(object, object)
    def _on_request_auto_review(self, _first_frame: object, auto_result: object) -> None:
        if self._worker is None:
            return
        result = auto_result
        if not isinstance(result, AutoMidlineResult):
            self._worker.set_auto_response(False)
            return
        dlg = AutoReviewDialog(
            overlay=result.overlay_image,
            title=f"Auto midline ({result.method}, conf={result.confidence:.2f})",
            parent=self,
        )
        approved = dlg.exec() == QDialog.Accepted
        self._worker.set_auto_response(approved)

    @Slot(object, float)
    def _on_request_manual_trace(self, first_frame: object, spacing_px: float) -> None:
        if self._worker is None:
            return
        try:
            frame = np.asarray(first_frame)
            dlg = ManualTraceDialog(image=frame, parent=self)
            if dlg.exec() != QDialog.Accepted:
                self._worker.set_manual_response(None)
                return
            raw_pts = dlg.points_image_xy()
            if len(raw_pts) < 2:
                self._worker.set_manual_response(None)
                return
            from .manual_input import resample_polyline_xy
            traced_xy = resample_polyline_xy(np.array(raw_pts, dtype=float), spacing_px=spacing_px)
            self._worker.set_manual_response(traced_xy)
        except Exception:
            self._worker.set_manual_response(None)

    @Slot(list)
    def _on_finished(self, results: list) -> None:
        ok_count = sum(1 for r in results if isinstance(r, PipelineResult) and r.status == "ok")
        fail_count = len(results) - ok_count
        self.status_label.setText(f"Done. OK: {ok_count}, Failed: {fail_count}")
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self._refresh_summary()

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self.status_label.setText(f"Failed: {message}")
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Pipeline failed", message)

    @Slot()
    def _cleanup_worker(self) -> None:
        self._worker = None
        self._worker_thread = None

    def _refresh_summary(self) -> None:
        output_root = self.output_edit.text().strip()
        profiles = discover_output_profiles(output_root)
        if not profiles:
            self.summary.setPlainText("No valid output profiles found yet.")
            return
        lines: list[str] = []
        for p in profiles:
            peak = f"{p.regr_peak:.5g}" if p.regr_peak is not None else "n/a"
            peak_loc = f"{p.regr_peak_location:.5g}" if p.regr_peak_location is not None else "n/a"
            lines.append(
                f"- {p.dataset_id}\n"
                f"  mode={p.initialization_mode}, points={len(p.l_domain)}, REGR_peak={peak}, REGR_peak_loc={peak_loc}\n"
                f"  output={p.output_dir}\n"
            )
        self.summary.setPlainText("\n".join(lines))


def _with_browse_button(line_edit: QLineEdit, callback) -> QWidget:
    wrap = QWidget()
    row = QGridLayout(wrap)
    row.setContentsMargins(0, 0, 0, 0)
    row.setColumnStretch(0, 1)
    row.addWidget(line_edit, 0, 0)
    btn = QPushButton("Browse")
    btn.clicked.connect(callback)
    row.addWidget(btn, 0, 1)
    return wrap


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize any numeric array to uint8 [0, 255], handling 16-bit and float."""
    a = arr.astype(np.float64)
    lo, hi = a.min(), a.max()
    if hi <= 1.0:
        a = a * 255.0
    elif hi > 255.0:
        a = (a - lo) / (hi - lo) * 255.0 if hi > lo else a * 0.0
    return np.clip(a, 0, 255).astype(np.uint8)


def _np_to_qimage(arr: np.ndarray) -> QImage:
    img = np.asarray(arr)
    if img.ndim == 2:
        h, w = img.shape
        img_u8 = np.ascontiguousarray(_normalize_to_uint8(img))
        qimg = QImage(img_u8.data, w, h, w, QImage.Format_Grayscale8)
        return qimg.copy()
    if img.ndim == 3 and img.shape[2] == 3:
        h, w, _ = img.shape
        rgb = np.ascontiguousarray(_normalize_to_uint8(img))
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return qimg.copy()
    if img.ndim == 3 and img.shape[2] == 4:
        h, w, _ = img.shape
        rgba = np.ascontiguousarray(_normalize_to_uint8(img))
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
        return qimg.copy()
    raise ValueError(f"Unsupported image shape for display: {img.shape}")


def launch() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    launch()
