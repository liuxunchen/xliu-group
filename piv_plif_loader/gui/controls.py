from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QComboBox, QCheckBox, QGroupBox,
    QSpinBox, QFormLayout
)
from PyQt6.QtCore import Qt

class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # --- 数据加载 ---
        load_group = QGroupBox("数据加载")
        load_layout = QVBoxLayout()
        self.btn_load_piv = QPushButton("加载 PIV 序列")
        self.btn_load_plif = QPushButton("加载 PLIF 序列 (可选)")
        load_layout.addWidget(self.btn_load_piv)
        load_layout.addWidget(self.btn_load_plif)

        bin_layout = QHBoxLayout()
        bin_layout.addWidget(QLabel("Bin 大小:"))
        self.spin_bin = QSpinBox()
        self.spin_bin.setRange(1, 8)
        self.spin_bin.setValue(1)
        bin_layout.addWidget(self.spin_bin)
        load_layout.addLayout(bin_layout)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        # --- 帧控制 ---
        frame_group = QGroupBox("帧控制")
        frame_layout = QVBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setValue(0)
        self.btn_play = QPushButton("播放")
        self.btn_stop = QPushButton("暂停")
        self.lbl_frame = QLabel("帧: 0 / 0")
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_stop)
        frame_layout.addWidget(self.slider)
        frame_layout.addLayout(btn_layout)
        frame_layout.addWidget(self.lbl_frame)
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)

        # --- 显示设置 ---
        display_group = QGroupBox("显示设置")
        display_layout = QVBoxLayout()

        self.combo_quantity = QComboBox()
        self.combo_quantity.addItems([
            'Velocity magnitude',
            'Vorticity',
            'Divergence',
            'Grad U',
            'Grad V',
            'Liutex'
        ])
        display_layout.addWidget(QLabel("选择标量场:"))
        display_layout.addWidget(self.combo_quantity)

        self.cb_quiver = QCheckBox("显示速度矢量 (quiver)")
        self.cb_streamline = QCheckBox("显示流线 (streamline)")
        self.cb_plif_overlay = QCheckBox("叠加 PLIF")
        self.cb_plif_overlay.setEnabled(False)
        display_layout.addWidget(self.cb_quiver)
        display_layout.addWidget(self.cb_streamline)
        display_layout.addWidget(self.cb_plif_overlay)

        # Quiver 箭头缩放
        quiver_layout = QHBoxLayout()
        quiver_layout.addWidget(QLabel("箭头缩放:"))
        self.quiver_scale = QSlider(Qt.Orientation.Horizontal)
        self.quiver_scale.setRange(1, 200)
        self.quiver_scale.setValue(50)
        quiver_layout.addWidget(self.quiver_scale)
        display_layout.addLayout(quiver_layout)

        # PLIF 参数（折叠在显示设置内）
        plif_param_layout = QFormLayout()
        self.spin_sigma_s = QSpinBox()
        self.spin_sigma_s.setRange(1, 50)
        self.spin_sigma_s.setValue(10)
        self.spin_sigma_r = QSpinBox()
        self.spin_sigma_r.setRange(1, 50)
        self.spin_sigma_r.setValue(10)
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 255)
        self.spin_threshold.setValue(30)
        self.spin_blur = QSpinBox()
        self.spin_blur.setRange(1, 30)
        self.spin_blur.setValue(15)
        plif_param_layout.addRow("sigma_s", self.spin_sigma_s)
        plif_param_layout.addRow("sigma_r", self.spin_sigma_r)
        plif_param_layout.addRow("阈值", self.spin_threshold)
        plif_param_layout.addRow("模糊", self.spin_blur)
        display_layout.addLayout(plif_param_layout)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # --- FFT 选点 ---
        fft_group = QGroupBox("单点 FFT 分析")
        fft_layout = QVBoxLayout()
        self.btn_pick_point = QPushButton("在流场上选点")
        fft_layout.addWidget(self.btn_pick_point)
        fft_group.setLayout(fft_layout)
        layout.addWidget(fft_group)

        layout.addStretch()
        self.setLayout(layout)