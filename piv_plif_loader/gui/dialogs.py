from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                             QWidget, QSlider, QFileDialog, QSpinBox,
                             QDoubleSpinBox, QPushButton, QFormLayout,
                             QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pathlib
import numpy as np
from core.piv_loader import load_piv
from core.plif_loader import load_plif, parse_cihx
from core.liutex import liutex_2d
from core.field_calculations import compute_all_fields


class PLIFCalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PLIF 标定参数")
        self.setMinimumWidth(350)
        layout = QVBoxLayout()
        info = QLabel("输入 TIFF 图像的物理标定参数。\n每个像素对应的物理尺寸 (dx, dy) 以及图像左上角的物理坐标 (x0, y0)。")
        info.setWordWrap(True)
        layout.addWidget(info)
        params_group = QGroupBox("物理标定")
        form = QFormLayout()
        self.spin_dx = QDoubleSpinBox(); self.spin_dx.setRange(1e-6, 1e6); self.spin_dx.setValue(1.0)
        self.spin_dx.setDecimals(6); self.spin_dx.setSingleStep(0.1)
        form.addRow("dx:", self.spin_dx)
        self.spin_dy = QDoubleSpinBox(); self.spin_dy.setRange(1e-6, 1e6); self.spin_dy.setValue(1.0)
        self.spin_dy.setDecimals(6); self.spin_dy.setSingleStep(0.1)
        form.addRow("dy:", self.spin_dy)
        self.spin_x0 = QDoubleSpinBox(); self.spin_x0.setRange(-1e6, 1e6); self.spin_x0.setValue(0.0)
        self.spin_x0.setDecimals(6); form.addRow("x0:", self.spin_x0)
        self.spin_y0 = QDoubleSpinBox(); self.spin_y0.setRange(-1e6, 1e6); self.spin_y0.setValue(0.0)
        self.spin_y0.setDecimals(6); form.addRow("y0:", self.spin_y0)
        self.spin_bin = QSpinBox(); self.spin_bin.setRange(1, 8); self.spin_bin.setValue(1)
        form.addRow("Bin:", self.spin_bin)
        params_group.setLayout(form); layout.addWidget(params_group)
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("确定"); btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("取消"); btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch(); btn_layout.addWidget(btn_ok); btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout); self.setLayout(layout)

    def get_params(self):
        return {'dx': self.spin_dx.value(), 'dy': self.spin_dy.value(),
                'x0': self.spin_x0.value(), 'y0': self.spin_y0.value(), 'size': self.spin_bin.value()}


class FFTDialog(QDialog):
    def __init__(self, freqs, amplitude, title="FFT 结果", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title); self.resize(650, 450)
        layout = QVBoxLayout()
        max_idx = np.argmax(amplitude) if len(amplitude) > 0 else 0
        peak_freq = freqs[max_idx] if len(freqs) > 0 else 0
        layout.addWidget(QLabel(f"峰值频率: {peak_freq:.4f} Hz   |   幅值: {amplitude[max_idx]:.4f}"))
        fig = Figure(figsize=(5, 4), dpi=100); ax = fig.add_subplot(111)
        ax.plot(freqs, amplitude, 'b-')
        if peak_freq > 0:
            ax.plot(peak_freq, amplitude[max_idx], 'ro', markersize=6, label=f'峰值: {peak_freq:.4f} Hz')
            ax.legend()
        ax.set_xlim(0, freqs[-1] if len(freqs) > 0 else 1)
        ax.set_xlabel('频率 (Hz)'); ax.set_ylabel('幅值'); ax.set_title('单边幅值谱'); ax.grid(True)
        layout.addWidget(FigureCanvas(fig)); self.setLayout(layout)


class ImageOverlayDialog(QDialog):
    """双图叠加 - 分别加载 PIV(txt/dat) 或 PLIF，叠加显示。"""

    QUANTITIES = ['Velocity magnitude', 'Vorticity', 'Divergence', 'Grad U', 'Grad V', 'Liutex']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像叠加"); self.resize(1100, 620)
        self.data = {'a': None, 'b': None}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        ctrl = QHBoxLayout()
        ctrl.addWidget(self._make_panel('a', "图像 A"))
        ctrl.addWidget(self._make_panel('b', "图像 B"))
        layout.addLayout(ctrl)
        self.fig, (self.ax_a, self.ax_b, self.ax_over) = plt.subplots(1, 3, figsize=(11.5, 3.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        bot = QHBoxLayout()
        bot.addWidget(QLabel("A 层透明度:"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(10, 90); self.alpha_slider.setValue(60)
        self.alpha_slider.valueChanged.connect(self._draw)
        bot.addWidget(self.alpha_slider)
        bot.addStretch()
        bot.addWidget(QPushButton("刷新", clicked=self._draw))
        layout.addLayout(bot)
        self.setLayout(layout)

    def _make_panel(self, side, title):
        grp = QGroupBox(title); ly = QVBoxLayout()
        combo_type = QComboBox()
        combo_type.addItems(["PIV 速度场 (txt/dat)", "PLIF 图像"])
        combo_type.currentTextChanged.connect(lambda _, s=side: self._on_type_change(s))
        setattr(self, f'combo_type_{side}', combo_type); ly.addWidget(combo_type)

        combo_plif = QComboBox()
        combo_plif.addItems(["TIFF 文件", "原始文件夹"]); combo_plif.setVisible(False)
        combo_plif.currentTextChanged.connect(lambda _, s=side: self._on_plif_sub_change(s))
        setattr(self, f"combo_plif_{side}", combo_plif); ly.addWidget(combo_plif)

        sf = QSpinBox(); sf.setMinimum(0); sf.setValue(0)
        sf.valueChanged.connect(self._draw); sf.setVisible(False)
        setattr(self, f"slider_frame_{side}", sf); ly.addWidget(sf)

        btn_load = QPushButton("加载文件...")
        btn_load.clicked.connect(lambda _, s=side: self._load_file(s))
        setattr(self, f'btn_load_{side}', btn_load); ly.addWidget(btn_load)

        lbl = QLabel("未加载"); lbl.setWordWrap(True)
        setattr(self, f'lbl_info_{side}', lbl); ly.addWidget(lbl)

        combo_qty = QComboBox()
        combo_qty.addItems(self.QUANTITIES); combo_qty.currentTextChanged.connect(self._draw)
        setattr(self, f'combo_qty_{side}', combo_qty); ly.addWidget(combo_qty)
        ly.addStretch(); grp.setLayout(ly); return grp

    def _on_type_change(self, side):
        txt = getattr(self, f"combo_type_{side}").currentText()
        cp = getattr(self, f"combo_plif_{side}")
        sf = getattr(self, f"slider_frame_{side}")
        qty = getattr(self, f"combo_qty_{side}")
        btn = getattr(self, f"btn_load_{side}")
        lbl = getattr(self, f"lbl_info_{side}")
        lbl.setText("未加载"); self.data[side] = None
        if txt.startswith("PIV"):
            cp.setVisible(False); sf.setVisible(False); qty.setVisible(True); btn.setText("加载文件...")
        else:
            cp.setVisible(True); sf.setVisible(False); qty.setVisible(False)
            self._on_plif_sub_change(side)
        self._draw()

    def _on_plif_sub_change(self, side):
        txt = getattr(self, f"combo_plif_{side}").currentText()
        sf = getattr(self, f"slider_frame_{side}")
        btn = getattr(self, f"btn_load_{side}")
        if txt == "原始文件夹": sf.setVisible(True); btn.setText("选择文件夹...")
        else: sf.setVisible(False); btn.setText("加载文件...")

    def _load_file(self, side):
        txt = getattr(self, f'combo_type_{side}').currentText()
        lbl = getattr(self, f'lbl_info_{side}')
        if txt.startswith("PIV"):
            path, _ = QFileDialog.getOpenFileName(self, "选择 PIV 文件", "", "PIV 文件 (*.txt *.dat);;所有文件 (*)")
            if not path: return
            try:
                with open(path, 'r') as f: fl = f.readline()
                ftype = 'dat' if ('I=' in fl or 'J=' in fl) else 'txt'
                piv = load_piv(path, filetype=ftype)
                X, Y, U, V = piv['X'], piv['Y'], piv['U'], piv['V']
                if X.shape[1] > 1 and np.allclose(X[0, :], X[0, 0]): X, Y, U, V = X.T, Y.T, U.T, V.T
                dx = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0
                dy = float(Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 1.0
                fields = compute_all_fields(U, V, dx, dy)
                R, _, _, _ = liutex_2d(U, V, dx, dy, signed=True); fields['Liutex'] = R
                self.data[side] = {'type': 'piv', 'X': X, 'Y': Y, 'fields': fields}
                lbl.setText(f"PIV: {path}\n{X.shape[1]}x{X.shape[0]} grid")
            except Exception as e:
                QMessageBox.critical(self, "加载 PIV 失败", str(e)); return
        else:
            sub = getattr(self, f"combo_plif_{side}").currentText()
            if sub.startswith("TIFF"):
                path, _ = QFileDialog.getOpenFileName(self, "选择 PLIF TIFF", "", "TIFF 文件 (*.tif *.tiff);;所有文件 (*)")
                if not path: return
                try:
                    plif = load_plif(path, size=1, x0=0, y0=0, dx=1, dy=1)
                    X1d, Y1d = plif['X'], plif['Y']; X, Y = np.meshgrid(X1d, Y1d)
                    s = plif['scalar']
                    if s.shape != X.shape: s = s.T if s.T.shape == X.shape else s
                    self.data[side] = {'type': 'plif', 'X': X, 'Y': Y, 'scalar': s}
                    lbl.setText(f"PLIF: {path}\n{X.shape[1]}x{X.shape[0]} px")
                except Exception as e:
                    QMessageBox.critical(self, "加载 PLIF 失败", str(e)); return
            else:
                folder = QFileDialog.getExistingDirectory(self, "选择 PLIF 原始文件夹")
                if not folder: return
                try:
                    import pathlib as _pl
                    fld = _pl.Path(folder)
                    cihx_files = list(fld.glob("*.cihx"))
                    shift = 0
                    if cihx_files:
                        meta = parse_cihx(cihx_files[0])
                        if meta["effective_side"].lower() == "higher" and meta["effective_depth"] > 0:
                            shift = meta["bit_depth"] - meta["effective_depth"]
                    tiff_files = sorted(list(fld.glob("*.tif")) + list(fld.glob("*.tiff")))
                    if not tiff_files:
                        raise FileNotFoundError("文件夹中没有 TIFF 文件")
                    sf = getattr(self, f"slider_frame_{side}")
                    sf.setMaximum(len(tiff_files) - 1); sf.setValue(0)
                    self.data[side] = {"type": "plif_raw",
                        "tiff_files": [str(f) for f in tiff_files],
                        "bit_shift": shift, "_cache": None}
                    lbl.setText(f"PLIF Raw: {folder}\n{len(tiff_files)} frames")
                except Exception as e:
                    QMessageBox.critical(self, "加载 PLIF 失败", str(e)); return
        self._draw()

    def _get_image(self, side):
        d = self.data[side]
        if d is None: return None
        if d['type'] == 'piv':
            qty = getattr(self, f'combo_qty_{side}').currentText()
            s = d['fields'].get(qty)
            return (d['X'], d['Y'], s) if s is not None else None
        elif d['type'] == 'plif':
            return d['X'], d['Y'], d['scalar']
        else:  # plif_raw — 按需加载单帧
            idx = getattr(self, f"slider_frame_{side}").value()
            cache = d.get("_cache")
            if cache is not None and cache[0] == idx:
                return cache[1], cache[2], cache[3]
            path = d["tiff_files"][idx]
            plif = load_plif(path, bit_shift=d["bit_shift"], size=1, x0=0, y0=0, dx=1, dy=1)
            X1d, Y1d = plif["X"], plif["Y"]
            X, Y = np.meshgrid(X1d, Y1d)
            s = plif["scalar"]
            if s.shape != X.shape:
                s = s.T if s.T.shape == X.shape else s
            d["_cache"] = (idx, X, Y, s)
            return X, Y, s

    def _draw(self, *args):
        img_a = self._get_image('a')
        img_b = self._get_image('b')
        for ax, img, lbl in [(self.ax_a, img_a, 'A'), (self.ax_b, img_b, 'B')]:
            ax.clear()
            if img is None: ax.set_title(f"图像 {lbl}: 待加载"); continue
            X, Y, s = img
            ax.imshow(s, cmap='jet', origin='lower', aspect='equal')
            sid = 'a' if lbl == 'A' else 'b'
            d = self.data[sid]
            if d is not None and d['type'] == 'piv':
                ax.set_title(f"图像 {lbl}: {getattr(self, f'combo_qty_{sid}').currentText()}")
            else:
                ax.set_title(f"图像 {lbl}: PLIF")

        self.ax_over.clear()
        if img_a is None or img_b is None:
            self.ax_over.set_title("叠加: 请加载两幅图像")
        else:
            alpha_a = self.alpha_slider.value() / 100.0
            _, _, sa = img_a; _, _, sb = img_b
            # 统一到相同尺寸
            target_shape = (max(sa.shape[0], sb.shape[0]), max(sa.shape[1], sb.shape[1]))
            import cv2
            if sa.shape != target_shape:
                sa = cv2.resize(sa.astype(np.float32), (target_shape[1], target_shape[0]))
            if sb.shape != target_shape:
                sb = cv2.resize(sb.astype(np.float32), (target_shape[1], target_shape[0]))
            # 归一化后 RGBA 混合
            na = np.nan_to_num(sa, nan=0.0); nb = np.nan_to_num(sb, nan=0.0)
            na_n = (na - np.nanmin(na)) / (np.nanmax(na) - np.nanmin(na) + 1e-12)
            nb_n = (nb - np.nanmin(nb)) / (np.nanmax(nb) - np.nanmin(nb) + 1e-12)
            rgba_a = plt.get_cmap("jet")(na_n); rgba_b = plt.get_cmap("hot")(nb_n)
            rgba_a[..., 3] = alpha_a; rgba_b[..., 3] = 1.0 - alpha_a
            blended = rgba_a.copy()
            mask = rgba_b[..., 3] > 0; blended[mask] = rgba_b[mask]
            self.ax_over.imshow(blended, origin="lower", aspect="equal")
            self.ax_over.set_title(f"叠加  (αA={alpha_a:.1f})")

        self.fig.subplots_adjust(left=0.05, right=0.98, bottom=0.12, top=0.9, wspace=0.3)
        self.canvas.draw()
