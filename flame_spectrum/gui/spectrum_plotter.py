# gui/spectrum_plotter.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

class SpectrumCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.toolbar = NavigationToolbar(self, parent)   # 工具栏
        self.axes.set_facecolor('#fcfcfc')

    def clear_all(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#fcfcfc')

    def plot_mixture(self, wavenumber, total_coef, individual_coefs, transmittance,
                     params=None, title="Mixture Spectrum",
                     grid_density='fine', show_wavelength=True):
        """
        绘制混合光谱：总吸收系数、各分子分量、透射率
        """
        self.clear_all()
        if len(wavenumber) == 0:
            self.draw()
            return

        # 下采样（避免 UI 卡顿）
        step = max(1, len(wavenumber) // 50000)
        wn = wavenumber[::step]

        # 颜色方案
        colors = ['orange', 'green', 'purple', 'brown', 'pink', 'cyan']
        # 分子列表按名称排序，保证颜色一致
        mol_names = sorted(individual_coefs.keys())

        # ---- 左轴：吸收系数 ----
        ax_left = self.axes
        lines = []
        labels = []

        # 各分子分量（虚线）
        for i, name in enumerate(mol_names):
            coef = individual_coefs[name][::step]
            if np.max(np.abs(coef)) > 1e-12:
                line, = ax_left.plot(wn, coef, linestyle='--', color=colors[i % len(colors)],
                                     linewidth=1.0, alpha=0.7, label=f'{name}')
                lines.append(line)
                labels.append(f'Abs. Coeff ({name})')

        # 总吸收系数（实线）
        if np.max(np.abs(total_coef)) > 1e-12:
            line_total, = ax_left.plot(wn, total_coef[::step], color='red',
                                       linewidth=1.5, alpha=0.8, label='Total Abs. Coeff')
            lines.append(line_total)
            labels.append('Total Abs. Coeff')

        ax_left.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=11)
        ax_left.set_ylabel('Absorption Coefficient (cm$^{-1}$)', color='red', fontsize=11)
        ax_left.tick_params(axis='y', labelcolor='red')

        # ---- 右轴：透射率 ----
        ax_right = ax_left.twinx()
        line_tr, = ax_right.plot(wn, transmittance[::step], color='blue',
                                 linewidth=1.5, alpha=0.9, label='Transmittance')
        lines.append(line_tr)
        labels.append('Transmittance')
        ax_right.set_ylabel('Transmittance', color='blue', fontsize=11)
        ax_right.tick_params(axis='y', labelcolor='blue')
        ax_right.set_ylim(-0.05, 1.05)

        # ---- 双 X 轴：波长 ----
        if show_wavelength:
            ax_top = ax_left.twiny()
            ax_top.set_xlim(ax_left.get_xlim())
            ax_top.set_xlabel('Wavelength (µm)', fontsize=11)
            ticks = ax_left.get_xticks()
            ticks = ticks[(ticks > 0) & (ticks >= wavenumber.min()) & (ticks <= wavenumber.max())]
            if len(ticks) > 10:
                ticks = np.linspace(wavenumber.min(), wavenumber.max(), 8)
            ax_top.set_xticks(ticks)
            ax_top.set_xticklabels([f'{10000/t:.3f}' for t in ticks])
            ax_top.tick_params(axis='x', direction='in', pad=10)

        # ---- 标题 ----
        final_title = title
        if params:
            T = params.get('T', '?')
            p = params.get('p', '?')
            l = params.get('l', '?')
            final_title += f" (T={T} K, p={p} atm, L={l} cm)"
        ax_left.set_title(final_title, fontsize=12, fontweight='bold')

        # ---- 图例 ----
        if lines:
            legend = ax_left.legend(lines, labels, loc='upper right', fontsize=9,
                                    framealpha=0.9)
            legend.get_frame().set_edgecolor('gray')

        # ---- 网格 ----
        grid_styles = {'fine': {'alpha':0.3, 'ls':':', 'lw':0.5},
                       'medium': {'alpha':0.4, 'ls':'--', 'lw':0.7},
                       'coarse': {'alpha':0.5, 'ls':'-', 'lw':0.8}}
        gs = grid_styles.get(grid_density, grid_styles['fine'])
        ax_left.grid(True, alpha=gs['alpha'], linestyle=gs['ls'], linewidth=gs['lw'])

        self.fig.tight_layout(pad=3.0)
        self.draw()