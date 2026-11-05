from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QComboBox, QCheckBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QFormLayout
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
        self.btn_load_plif_raw = QPushButton("加载 PLIF 原始文件夹")
        load_layout.addWidget(self.btn_load_plif_raw)

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
        display_layout.addWidget(self.cb_quiver)
        display_layout.addWidget(self.cb_streamline)

        # Quiver 箭头缩放
        quiver_scale_layout = QHBoxLayout()
        quiver_scale_layout.addWidget(QLabel("箭头缩放:"))
        self.quiver_scale = QDoubleSpinBox()
        self.quiver_scale.setRange(0.01, 1000.0)
        self.quiver_scale.setValue(50.0)
        self.quiver_scale.setSingleStep(5.0)
        self.quiver_scale.setDecimals(1)
        quiver_scale_layout.addWidget(self.quiver_scale)
        display_layout.addLayout(quiver_scale_layout)


        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # --- FFT 选点 ---
        fft_group = QGroupBox("单点 FFT 分析")
        fft_layout = QVBoxLayout()
        self.btn_pick_point = QPushButton("在流场上选点")
        fft_layout.addWidget(self.btn_pick_point)
        self.btn_overlay = QPushButton("图像叠加")
        fft_layout.addWidget(self.btn_overlay)
        fft_group.setLayout(fft_layout)
        layout.addWidget(fft_group)

        layout.addStretch()
        self.setLayout(layout)
