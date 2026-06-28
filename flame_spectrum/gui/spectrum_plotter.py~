import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Compatible Qt backend import (Supports PyQt5, PyQt6, PySide2, PySide6)
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Set universal, safe fonts to avoid missing CJK font warnings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class SpectrumCanvas(FigureCanvas):
    """
    A reusable Matplotlib canvas for plotting HITRAN/Flame spectra (Qt compatible).
    Optimized for high-resolution line plots with dual Y-axes and dual X-axes.
    """
    
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
        # Clean default style
        self.axes.set_facecolor('#fcfcfc')
        self.fig.set_facecolor('#ffffff')

    def clear_all(self):
        """Clear all axes and reset the canvas."""
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#fcfcfc')

    def plot_spectrum(self, wavenumber, coef, transmittance, 
                      params=None, title="Spectrum",
                      show_abs=True, show_trans=True,
                      grid_density='fine', show_wavelength=True):
        """
        Plot the spectrum using high-performance line plots.
        
        Parameters:
        -----------
        wavenumber : np.ndarray
            Wavenumber array (cm⁻¹)
        coef : np.ndarray
            Absorption coefficient array (cm⁻¹)
        transmittance : np.ndarray
            Transmittance/Absorptance array (0 to 1)
        params : dict, optional
            Calculation parameters (T, p, l, c) for the title
        title : str
            Main title of the plot
        show_abs : bool
            Whether to show absorption coefficient
        show_trans : bool
            Whether to show transmittance
        grid_density : str
            'fine', 'medium', or 'coarse'
        show_wavelength : bool
            Whether to show the secondary top x-axis (Wavelength in µm)
        """
        self.clear_all()
        
        if len(wavenumber) == 0:
            self.draw()
            return

        # Grid styling
        grid_styles = {
            'fine':   {'alpha': 0.3, 'ls': ':',  'lw': 0.5},
            'medium': {'alpha': 0.4, 'ls': '--', 'lw': 0.7},
            'coarse': {'alpha': 0.5, 'ls': '-',  'lw': 0.8}
        }
        grid_cfg = grid_styles.get(grid_density, grid_styles['fine'])

        ax_main = self.axes
        lines = []
        labels = []

        # --- 1. Plot Absorption Coefficient (Left Y-Axis) ---
        if show_abs:
            color_abs = '#1f77b4'  # Matplotlib default blue
            
            # Downsample if data points are too massive (>500k) to keep UI responsive
            step = max(1, len(wavenumber) // 500000)
            wn_plot = wavenumber[::step]
            coef_plot = coef[::step]
            
            line_abs, = ax_main.plot(wn_plot, coef_plot, color=color_abs, 
                                     linewidth=0.8, linestyle='-', 
                                     label='Absorption Coefficient')
            lines.append(line_abs)
            labels.append('Absorption Coefficient (cm⁻¹)')
                
            ax_main.set_ylabel('Absorption Coefficient (cm⁻¹)', color=color_abs, fontsize=11)
            ax_main.tick_params(axis='y', labelcolor=color_abs, labelsize=10)

        # --- 2. Plot Transmittance / Absorptance (Right Y-Axis) ---
        if show_trans:
            color_trans = '#ff7f0e'  # Matplotlib default orange
            if show_abs:
                ax_trans = ax_main.twinx()
            else:
                ax_trans = ax_main
                
            step = max(1, len(wavenumber) // 500000)
            wn_plot = wavenumber[::step]
            trans_plot = transmittance[::step]
                
            line_trans, = ax_trans.plot(wn_plot, trans_plot, color=color_trans, 
                                        linewidth=0.8, linestyle='-', alpha=0.9,
                                        label='Transmittance')
            lines.append(line_trans)
            labels.append('Transmittance')
            
            ax_trans.set_ylim(-0.05, 1.05) 
            ax_trans.set_ylabel('Transmittance', color=color_trans, fontsize=11)
            ax_trans.tick_params(axis='y', labelcolor=color_trans, labelsize=10)

        # --- 3. Dual X-Axis (Bottom: Wavenumber, Top: Wavelength) ---
        ax_main.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11, labelpad=10)
        ax_main.tick_params(axis='x', labelsize=10)
        
        if show_wavelength:
            ax_top = ax_main.twiny()
            ax_top.set_xlabel('Wavelength (µm)', fontsize=11, labelpad=10)
            
            # Sync limits
            x_min, x_max = ax_main.get_xlim()
            ax_top.set_xlim(x_min, x_max)
            
            # Generate smart ticks for wavelength
            x_ticks = ax_main.get_xticks()
            x_ticks = x_ticks[(x_ticks >= wavenumber.min()) & (x_ticks <= wavenumber.max())]
            
            # Limit number of ticks to avoid crowding
            if len(x_ticks) > 10:
                step = max(1, len(x_ticks) // 8)
                x_ticks = x_ticks[::step]
                
            x_ticks = x_ticks[x_ticks > 0] # Avoid division by zero
            
            if len(x_ticks) > 0:
                wavelength_ticks = 10000.0 / x_ticks
                ax_top.set_xticks(x_ticks)
                ax_top.set_xticklabels([f'{w:.3f}' for w in wavelength_ticks], fontsize=10)
            
            ax_top.tick_params(axis='x', direction='in', pad=10)

        # --- 4. Title and Legend ---
        final_title = title
        if params:
            T = params.get('T', '?')
            p = params.get('p', '?')
            l = params.get('l', '?')
            final_title += f"\n(T={T} K, p={p} atm, L={l} cm)"
                
        ax_main.set_title(final_title, fontsize=12, fontweight='bold', pad=20)

        if lines:
            # Place legend in the main axis
            legend = ax_main.legend(lines, labels, loc='upper right', fontsize=10, framealpha=0.9)
            legend.get_frame().set_edgecolor('gray')
            legend.get_frame().set_linewidth(0.5)

        # --- 5. Grid and Layout ---
        ax_main.grid(True, alpha=grid_cfg['alpha'], linestyle=grid_cfg['ls'], 
                     linewidth=grid_cfg['lw'], zorder=0)
        
        self.fig.tight_layout(pad=3.0)
        self.draw()
