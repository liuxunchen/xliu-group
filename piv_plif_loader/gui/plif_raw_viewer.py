#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLIF 原始图像浏览与画图组件。
提供独立的 PLIF 图像序列浏览、统计信息与 FFT 分析。
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QGroupBox, QFormLayout,
    QCheckBox, QFileDialog, QMessageBox, QWidget
)
from PyQt6.QtCore import Qt, QTimer
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.plif_loader import load_plif_raw_folder, compute_plif_statistics
from core.fft_analyzer import point_fft
from gui.dialogs import FFTDialog


class PlifRawViewerDialog(QDialog):
    """PLIF 原始图像序列浏览对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PLIF 原始图像浏览")
        self.resize(1100, 750)

        self.frames = []            # list of dict from load_plif
        self.metadata = None
        self.current_frame = 0
        self.bit_shift_override = None  # None = auto-detect
        self.playing = False
        self.stats = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.cbar = None
        self.pick_mode = False
        self._pick_connection = None

        self.init_ui()

    def init_ui(self):
        main = QHBoxLayout()

        # ---- 左侧: 图像 ----
        left = QVBoxLayout()

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_img = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        left.addWidget(self.canvas)

        # 帧控制
        frame_ctrl = QHBoxLayout()
        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self.start_play)
        self.btn_stop = QPushButton("暂停")
        self.btn_stop.clicked.connect(self.stop_play)
        self.btn_prev = QPushButton("上一帧")
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next = QPushButton("下一帧")
        self.btn_next.clicked.connect(self.next_frame)
        frame_ctrl.addWidget(self.btn_play)
        frame_ctrl.addWidget(self.btn_stop)
        frame_ctrl.addWidget(self.btn_prev)
        frame_ctrl.addWidget(self.btn_next)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.set_frame)
        self.lbl_frame = QLabel("帧: 0 / 0")

        left.addLayout(frame_ctrl)
        left.addWidget(self.slider)
        left.addWidget(self.lbl_frame)

        # ---- 右侧: 信息面板 ----
        right = QVBoxLayout()

        # 加载控制
        load_group = QGroupBox("数据加载")
        load_layout = QVBoxLayout()
        self.btn_load_folder = QPushButton("加载 PLIF 原始文件夹")
        self.btn_load_folder.clicked.connect(self.load_folder)
        load_layout.addWidget(self.btn_load_folder)
        self.lbl_folder = QLabel("未加载")
        self.lbl_folder.setWordWrap(True)
        load_layout.addWidget(self.lbl_folder)

        self.cb_auto_shift = QCheckBox("自动位深检测 (cihx)")
        self.cb_auto_shift.setChecked(True)
        load_layout.addWidget(self.cb_auto_shift)
        load_group.setLayout(load_layout)
        right.addWidget(load_group)

        # 元数据
        meta_group = QGroupBox("相机元数据")
        self.meta_form = QFormLayout()
        self.lbl_meta_camera = QLabel("-")
        self.lbl_meta_res = QLabel("-")
        self.lbl_meta_rate = QLabel("-")
        self.lbl_meta_shutter = QLabel("-")
        self.lbl_meta_depth = QLabel("-")
        self.lbl_meta_frames = QLabel("-")
        self.meta_form.addRow("相机:", self.lbl_meta_camera)
        self.meta_form.addRow("分辨率:", self.lbl_meta_res)
        self.meta_form.addRow("帧率:", self.lbl_meta_rate)
        self.meta_form.addRow("快门:", self.lbl_meta_shutter)
        self.meta_form.addRow("位深:", self.lbl_meta_depth)
        self.meta_form.addRow("总帧数:", self.lbl_meta_frames)
        meta_group.setLayout(self.meta_form)
        right.addWidget(meta_group)

        # 当前帧统计
        stats_group = QGroupBox("当前帧统计")
        self.stats_form = QFormLayout()
        self.lbl_stat_min = QLabel("-")
        self.lbl_stat_max = QLabel("-")
        self.lbl_stat_mean = QLabel("-")
        self.lbl_stat_std = QLabel("-")
        self.stats_form.addRow("最小值:", self.lbl_stat_min)
        self.stats_form.addRow("最大值:", self.lbl_stat_max)
        self.stats_form.addRow("均值:", self.lbl_stat_mean)
        self.stats_form.addRow("标准差:", self.lbl_stat_std)
        stats_group.setLayout(self.stats_form)
        right.addWidget(stats_group)

        # 保存按钮
        self.btn_save = QPushButton("保存当前帧为 PNG")
        self.btn_save.clicked.connect(self.save_current_frame)
        self.btn_save.setEnabled(False)
        right.addWidget(self.btn_save)

        # 单点 FFT 选点
        fft_group = QGroupBox("单点 FFT 分析")
        fft_layout = QVBoxLayout()
        self.btn_pick_plif = QPushButton("在 PLIF 图像上选点")
        self.btn_pick_plif.clicked.connect(self.enter_pick_mode)
        self.btn_pick_plif.setEnabled(False)
        fft_layout.addWidget(self.btn_pick_plif)
        fft_group.setLayout(fft_layout)
        right.addWidget(fft_group)

        right.addStretch()
        main.addLayout(left, 3)
        main.addLayout(right, 1)
        self.setLayout(main)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择 PLIF 原始数据文件夹")
        if not folder:
            return

        try:
            bit_shift = None if self.cb_auto_shift.isChecked() else 0
            result = load_plif_raw_folder(folder, bit_shift=bit_shift, verbose=True)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
            return

        self.frames = result['frames']
        self.metadata = result['metadata']
        self.stats = compute_plif_statistics(self.frames)
        self.current_frame = 0

        self.slider.setMaximum(len(self.frames) - 1)
        self.slider.setValue(0)
        self.lbl_folder.setText(folder)
        self.btn_save.setEnabled(True)
        self.btn_pick_plif.setEnabled(True)

        # 更新元数据显示
        if self.metadata:
            self.lbl_meta_camera.setText(self.metadata.get('camera_name', '-'))
            self.lbl_meta_res.setText(f"{self.metadata.get('width', '-')}x{self.metadata.get('height', '-')}")
            self.lbl_meta_rate.setText(f"{self.metadata.get('record_rate', '-')} fps")
            self.lbl_meta_shutter.setText(f"1/{self.metadata.get('shutter_speed', '-')} s")
            self.lbl_meta_depth.setText(
                f"{self.metadata.get('effective_depth', '-')} bit "
                f"({self.metadata.get('effective_side', '-')} side)"
            )
            self.lbl_meta_frames.setText(str(self.metadata.get('total_frame', '-')))

        self.update_plot()

    def update_plot(self):
        if not self.frames:
            return
        idx = max(0, min(self.current_frame, len(self.frames) - 1))
        frame = self.frames[idx]
        scalar = frame['scalar']
        x_num = frame['x_num']
        y_num = frame['y_num']

        # 先移除旧 colorbar，再清除 axes（否则 colorbar 内部状态丢失）
        if self.cbar is not None:
            self.cbar.remove()
            self.cbar = None

        self.ax_img.clear()

        # 主图像
        vmin = np.percentile(scalar[scalar > 0], 2) if np.any(scalar > 0) else 0
        vmax = np.percentile(scalar[scalar > 0], 98) if np.any(scalar > 0) else 1
        im = self.ax_img.imshow(scalar, cmap='hot', aspect='equal',
                                origin='lower', vmin=vmin, vmax=vmax)

        self.cbar = self.fig.colorbar(im, ax=self.ax_img, label="Intensity")
        self.ax_img.set_title(f"PLIF 帧 {idx+1}/{len(self.frames)}  —  "
                              f"{frame['file_name']}")
        self.ax_img.set_xlabel(f"x ({x_num} px)")
        self.ax_img.set_ylabel(f"y ({y_num} px)")

        self.fig.tight_layout()
        self.canvas.draw()

        # 更新统计标签
        if self.stats and idx < len(self.stats['per_frame']):
            s = self.stats['per_frame'][idx]
            self.lbl_stat_min.setText(f"{s['min']:.2f}")
            self.lbl_stat_max.setText(f"{s['max']:.2f}")
            self.lbl_stat_mean.setText(f"{s['mean']:.2f}")
            self.lbl_stat_std.setText(f"{s['std']:.2f}")

        self.lbl_frame.setText(f"帧: {idx} / {len(self.frames)-1}")

    # ---- 帧控制 ----
    def start_play(self):
        self.playing = True
        self.timer.start(100)

    def stop_play(self):
        self.playing = False
        self.timer.stop()

    def next_frame(self):
        if not self.frames:
            return
        nxt = (self.current_frame + 1) % len(self.frames)
        self.slider.setValue(nxt)

    def prev_frame(self):
        if not self.frames:
            return
        prv = (self.current_frame - 1) % len(self.frames)
        self.slider.setValue(prv)

    def set_frame(self, idx):
        self.current_frame = idx
        self.update_plot()

    # ---- FFT 选点 ----
    def enter_pick_mode(self):
        if not self.frames:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return
        self.pick_mode = True
        self._pick_connection = self.canvas.mpl_connect(
            "button_press_event", self.on_pick_plif)
        self.btn_pick_plif.setText("点击图像选点... (再次点击图像取消)")

    def on_pick_plif(self, event):
        if not self.pick_mode or event.inaxes != self.ax_img:
            return
        self.pick_mode = False
        if self._pick_connection is not None:
            self.canvas.mpl_disconnect(self._pick_connection)
            self._pick_connection = None
        self.btn_pick_plif.setText("在 PLIF 图像上选点")

        ix = int(round(event.xdata))
        iy = int(round(event.ydata))
        if not (0 <= ix < self.frames[0]["x_num"] and 0 <= iy < self.frames[0]["y_num"]):
            QMessageBox.warning(self, "越界", f"坐标 ({ix}, {iy}) 超出图像范围")
            return

        signal = np.array([f["scalar"][iy, ix] for f in self.frames])
        freqs, amp = point_fft(signal, dt=1.0)
        dlg = FFTDialog(freqs, amp, title=f"PLIF FFT at ({ix}, {iy})")
        dlg.exec()


    def save_current_frame(self):
        if not self.frames:
            return
        idx = max(0, min(self.current_frame, len(self.frames) - 1))
        default_name = f"plif_frame_{idx:06d}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存当前帧", default_name,
            "PNG 图像 (*.png);;所有文件 (*)")
        if file_path:
            self.fig.savefig(file_path, dpi=150, bbox_inches='tight')
            QMessageBox.information(self, "保存成功", f"已保存至 {file_path}")
