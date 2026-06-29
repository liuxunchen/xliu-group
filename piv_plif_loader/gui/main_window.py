import sys
import pathlib
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QFileDialog,
    QHBoxLayout, QVBoxLayout, QWidget, QMessageBox,
    QPushButton, QLabel
)
from PyQt6.QtCore import Qt, QTimer

from gui.canvas import MplCanvas, NavigationToolbar
from gui.controls import ControlPanel
from gui.dialogs import FFTDialog
from core.piv_loader import load_piv_batch, load_piv
from core.plif_loader import load_plif_batch, load_plif, align_plif_to_piv, enhance_plif
from core.liutex import liutex_2d
from core.field_calculations import compute_all_fields
from core.fft_analyzer import point_fft


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PIV/PLIF Liutex 分析工具")
        self.resize(1200, 800)

        # 数据容器
        self.piv_data_list = []          # 每帧的字典
        self.plif_data_list = []         # 可能为空
        self.liutex_R = []               # Liutex 每帧
        self.derived_fields = []         # 派生场每帧
        self.current_frame = 0

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False

        # 选点状态
        self.pick_mode = False

        # colorbar 引用
        self.cbar = None

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()

        # ---- 左侧：画布 + 工具栏 + 保存按钮 ----
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # 画布
        self.canvas_flow = MplCanvas(self)
        # 工具栏（parent 设为 None 避免被父对象重设）
        self.toolbar_flow = NavigationToolbar(self.canvas_flow, None)
        left_layout.addWidget(self.toolbar_flow)
        left_layout.addWidget(self.canvas_flow)

        # 保存标量场数据按钮
        self.btn_save_scalar = QPushButton("保存当前标量场数据")
        self.btn_save_scalar.clicked.connect(self.save_scalar_data)
        left_layout.addWidget(self.btn_save_scalar)


        left_widget.setLayout(left_layout)

        # ---- 右侧控制面板 ----
        self.controls = ControlPanel()

        # 信号连接
        self.controls.btn_load_piv.clicked.connect(self.load_piv_data)
        self.controls.btn_load_plif.clicked.connect(self.load_plif_data)
        self.controls.btn_play.clicked.connect(self.start_play)
        self.controls.btn_stop.clicked.connect(self.stop_play)
        self.controls.slider.valueChanged.connect(self.set_frame)
        self.controls.btn_pick_point.clicked.connect(self.enter_pick_mode)
        self.controls.combo_quantity.currentTextChanged.connect(self.update_plot)
        self.controls.cb_quiver.stateChanged.connect(self.update_plot)
        self.controls.cb_streamline.stateChanged.connect(self.update_plot)
        self.controls.cb_plif_overlay.stateChanged.connect(self.update_plot)
        self.controls.spin_sigma_s.valueChanged.connect(self.update_plot)
        self.controls.spin_sigma_r.valueChanged.connect(self.update_plot)
        self.controls.spin_threshold.valueChanged.connect(self.update_plot)
        self.controls.spin_blur.valueChanged.connect(self.update_plot)
        self.controls.quiver_scale.valueChanged.connect(self.update_plot)

        main_layout.addWidget(left_widget, 3)
        main_layout.addWidget(self.controls, 1)
        central.setLayout(main_layout)

    # ========== 数据加载 ==========
    def load_piv_data(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择 PIV 文件", "", "文本文件 (*.txt *.dat);;所有文件 (*)")
        if not files:
            return
        # 自动判断格式
        try:
            with open(files[0], 'r') as f:
                first_line = f.readline()
            if 'I=' in first_line or 'J=' in first_line:
                ftype = 'dat'
            else:
                ftype = 'txt'
        except Exception:
            ftype = 'txt'

        bin_size = self.controls.spin_bin.value()
        try:
            self.piv_data_list = load_piv_batch(files, filetype=ftype, size=bin_size,
                                                verbose=True)
            if not self.piv_data_list:
                raise RuntimeError("没有成功加载任何文件")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
            return

        # 预计算 Liutex 与派生场
        self.precompute_fields()

        self.controls.slider.setMaximum(len(self.piv_data_list) - 1)
        self.controls.slider.setValue(0)
        self.controls.lbl_frame.setText(f"帧: 0 / {len(self.piv_data_list)-1}")
        self.controls.cb_plif_overlay.setEnabled(bool(self.plif_data_list))
        self.update_plot()

    def load_plif_data(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择 PLIF 文件", "", "TIFF 文件 (*.tif *.tiff);;所有文件 (*)")
        if not files:
            return
        # PLIF 使用默认物理坐标（可根据需要调整）
        try:
            self.plif_data_list = load_plif_batch(files, size=1, verbose=True)
        except Exception as e:
            QMessageBox.critical(self, "PLIF 加载失败", str(e))
            return
        self.controls.cb_plif_overlay.setEnabled(True)
        self.update_plot()

    def precompute_fields(self):
        """计算所有帧的 Liutex 及派生场，并确保坐标方向正确"""
        self.liutex_R = []
        self.derived_fields = []
        for piv in self.piv_data_list:
            X, Y = piv['X'], piv['Y']
            U, V = piv['U'], piv['V']

            # 转置为 (ny, nx) 以便梯度计算
            if X.shape[1] > 0 and np.allclose(X[0, :], X[0, 0]):
                X = X.T; Y = Y.T; U = U.T; V = V.T
                piv['X'], piv['Y'] = X, Y
                piv['U'], piv['V'] = U, V
                piv['xnum'], piv['ynum'] = piv['ynum'], piv['xnum']

            dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
            dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0

            R, _, _, _ = liutex_2d(U, V, dx, dy, signed=True)
            self.liutex_R.append(R)

            fields = compute_all_fields(U, V, dx, dy)
            self.derived_fields.append(fields)

    # ========== 帧控制 ==========
    def start_play(self):
        self.playing = True
        self.timer.start(100)

    def stop_play(self):
        self.playing = False
        self.timer.stop()

    def next_frame(self):
        if not self.piv_data_list:
            return
        nxt = (self.current_frame + 1) % len(self.piv_data_list)
        self.controls.slider.setValue(nxt)

    def set_frame(self, idx):
        self.current_frame = idx
        self.controls.lbl_frame.setText(f"帧: {idx} / {len(self.piv_data_list)-1}")
        self.update_plot()

    # ========== 绘图更新 ==========
    def update_plot(self):
        if not self.piv_data_list:
            return
        idx = self.current_frame
        piv = self.piv_data_list[idx]
        X, Y = piv['X'], piv['Y']
        U, V = piv['U'], piv['V']

        quantity = self.controls.combo_quantity.currentText()
        if quantity == 'Liutex':
            scalar = self.liutex_R[idx]
        else:
            scalar = self.derived_fields[idx].get(quantity,
                     self.derived_fields[idx].get('Velocity magnitude',
                         list(self.derived_fields[idx].values())[0]))

        ax = self.canvas_flow.ax

        # 移除旧 colorbar
        if self.cbar is not None:
            self.cbar.remove()
            self.cbar = None

        ax.clear()
        cf = ax.contourf(X, Y, scalar, levels=50, cmap='jet')
        self.cbar = self.canvas_flow.fig.colorbar(cf, ax=ax)

        # Quiver
        if self.controls.cb_quiver.isChecked():
            skip = max(1, X.shape[0] // 20)
            scale = self.controls.quiver_scale.value()
            ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                      U[::skip, ::skip], V[::skip, ::skip],
                      scale=scale, alpha=0.7)

        # Streamplot
        if self.controls.cb_streamline.isChecked():
            x_1d = X[0, :].astype('float64')
            y_1d = Y[:, 0].astype('float64')
            if x_1d[0] > x_1d[-1]:
                x_1d = x_1d[::-1]; U = U[:, ::-1]; V = V[:, ::-1]
            if y_1d[0] > y_1d[-1]:
                y_1d = y_1d[::-1]; U = U[::-1, :]; V = V[::-1, :]
            x_1d = np.linspace(x_1d[0], x_1d[-1], len(x_1d))
            y_1d = np.linspace(y_1d[0], y_1d[-1], len(y_1d))
            speed = np.sqrt(U**2 + V**2)
            ax.streamplot(x_1d, y_1d, U, V, color=speed, linewidth=1, cmap='gray')

        # PLIF 叠加
        if self.controls.cb_plif_overlay.isChecked() and self.plif_data_list:
            plif_idx = min(idx, len(self.plif_data_list) - 1)
            aligned = align_plif_to_piv(self.plif_data_list[plif_idx], piv)
            enhanced = enhance_plif(aligned,
                                    sigma_s=self.controls.spin_sigma_s.value(),
                                    sigma_r=self.controls.spin_sigma_r.value(),
                                    thres_hold=self.controls.spin_threshold.value(),
                                    blur_size=self.controls.spin_blur.value())
            ax.contourf(X, Y, enhanced, alpha=0.5, cmap='Reds',
                        levels=np.linspace(1, 110, 10))

        ax.set_aspect('equal')
        ax.set_title(quantity)
        self.canvas_flow.draw()

    # ========== FFT 选点 ==========
    def enter_pick_mode(self):
        if not self.piv_data_list:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return
        self.pick_mode = True
        self.canvas_flow.mpl_connect('button_press_event', self.on_pick)

    def on_pick(self, event):
        if not self.pick_mode or event.inaxes != self.canvas_flow.ax:
            return
        self.pick_mode = False
        x_click, y_click = event.xdata, event.ydata

        X = self.piv_data_list[0]['X']
        Y = self.piv_data_list[0]['Y']
        dist = np.sqrt((X - x_click)**2 + (Y - y_click)**2)
        iy, ix = np.unravel_index(np.argmin(dist), X.shape)
        print(f"选中点: ({X[iy,ix]:.3f}, {Y[iy,ix]:.3f})")

        signal = []
        for frame in range(len(self.piv_data_list)):
            signal.append(self.liutex_R[frame][iy, ix])

        freqs, amp = point_fft(np.array(signal), dt=1.0)
        dlg = FFTDialog(freqs, amp, title=f"FFT at ({X[iy,ix]:.2f}, {Y[iy,ix]:.2f})")
        dlg.exec()

    # ========== 保存标量场 ==========
    def save_scalar_data(self):
        if not self.piv_data_list:
            QMessageBox.warning(self, "警告", "没有数据可保存")
            return
        idx = self.current_frame
        quantity = self.controls.combo_quantity.currentText()
        if quantity == 'Liutex':
            data = self.liutex_R[idx]
        else:
            data = self.derived_fields[idx].get(quantity)
            if data is None:
                QMessageBox.warning(self, "错误", "无法获取当前标量场")
                return
        default_name = f"{quantity}_frame{idx}.npy"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存标量场数据", default_name,
            "NumPy 数组 (*.npy);;所有文件 (*)")
        if file_path:
            np.save(file_path, data)
            QMessageBox.information(self, "保存成功", f"数据已保存至 {file_path}")
