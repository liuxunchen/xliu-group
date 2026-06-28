# gui/components.py
import sys
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QGroupBox, QFileDialog, QMessageBox, QTextEdit, QCheckBox, QDoubleSpinBox, QSpinBox, QProgressBar, QSplitter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    """Matplotlib 画布，用于在 PyQt 中嵌入图形"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

class CalculationThread(QThread):
    """通用计算线程，执行光谱计算并返回结果"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, hitran, params):
        super().__init__()
        self.hitran = hitran
        self.params = params

    def run(self):
        try:
            OD, Ab, Tr, wavenumber, coef = self.hitran.OD(
                self.params['T'], self.params['p'], self.params['c'], self.params['l'],
                start=self.params['start'], end=self.params['end'],
                resolution=self.params['resolution'], omega_wing=self.params['omega_wing']
            )
            results = {
                'wavenumber': wavenumber,
                'coef': coef,
                'OD': OD,
                'Ab': Ab,
                'Tr': Tr,
                'params': self.params
            }
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class SpectrumCanvas(MplCanvas):
    """
    Enhanced canvas with bar-style absorption/transmittance plots
    and a secondary x-axis showing wavelength in micrometres.
    """
    def plot_spectrum(self, results, show_abs=True, show_trans=True,
                      grid_density='fine', show_wavelength=True):
        self.axes.clear()
        wavenumber = results['wavenumber']
        coef = results['coef']
        Tr = results['Tr']

        if len(wavenumber) > 1:
            bar_width = np.mean(np.diff(wavenumber)) * 0.8
        else:
            bar_width = 0.1

        # grid style
        if grid_density == 'fine':
            alpha, ls, lw = 0.2, ':', 0.5
        elif grid_density == 'medium':
            alpha, ls, lw = 0.3, '--', 0.7
        else:
            alpha, ls, lw = 0.4, '-', 0.8

        lines = []

        # absorption coefficient bars
        if show_abs:
            bars = self.axes.bar(wavenumber, coef,
                                 width=bar_width,
                                 color='tab:blue', alpha=0.7,
                                 edgecolor='tab:blue', linewidth=0.5,
                                 label='Absorption coefficient')
            if bars:
                lines.append(bars[0])

        # transmittance bars
        if show_trans:
            if show_abs:
                ax2 = self.axes.twinx()
                color = 'tab:orange'
                ax2.set_ylabel('Transmittance', color=color)
                bars_tr = ax2.bar(wavenumber, Tr,
                                  width=bar_width,
                                  color=color, alpha=0.5,
                                  edgecolor=color, linewidth=0.5,
                                  label='Transmittance')
                if bars_tr:
                    lines.append(bars_tr[0])
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim(0, 1)
            else:
                bars_tr = self.axes.bar(wavenumber, Tr,
                                        width=bar_width,
                                        color='tab:orange', alpha=0.5,
                                        edgecolor='tab:orange', linewidth=0.5,
                                        label='Transmittance')
                if bars_tr:
                    lines.append(bars_tr[0])
                self.axes.set_ylim(0, 1)

        # title and labels
        params = results.get('params', {})
        T = params.get('T', '?')
        p = params.get('p', '?')
        l = params.get('l', '?')
        c = params.get('c', '?')
        self.axes.set_title(f"HITRAN Spectrum\nT={T} K, p={p} atm, l={l} cm, c={c:.2e}")
        self.axes.set_xlabel('Wavenumber (cm⁻¹)')
        if show_abs:
            self.axes.set_ylabel('Absorption coefficient (cm⁻¹)', color='black')
        else:
            self.axes.set_ylabel('Transmittance', color='tab:orange')

        # secondary x-axis (wavelength)
        if show_wavelength:
            ax_top = self.axes.twiny()
            ax_top.set_xlabel('Wavelength (µm)')
            ax_top.set_xlim(self.axes.get_xlim())
            x_ticks = self.axes.get_xticks()
            x_min, x_max = wavenumber.min(), wavenumber.max()
            x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]
            if len(x_ticks) > 10:
                step = max(1, len(x_ticks) // 8)
                x_ticks = x_ticks[::step]
            if len(x_ticks) > 0:
                wavelength_ticks = 10000.0 / x_ticks
                ax_top.set_xticks(x_ticks)
                ax_top.set_xticklabels([f'{w:.3f}' for w in wavelength_ticks])
            ax_top.tick_params(axis='x', direction='in', pad=10)
            self._ax_top = ax_top

        if lines:
            labels = [line.get_label() for line in lines]
            self.axes.legend(lines, labels, loc='upper right')

        self.axes.grid(True, alpha=alpha, linestyle=ls, linewidth=lw)
        self.fig.tight_layout()
        self.draw()
            
