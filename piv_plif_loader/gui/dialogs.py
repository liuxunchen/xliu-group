from PyQt6.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class FFTDialog(QDialog):
    def __init__(self, freqs, amplitude, title="FFT 结果", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(600, 400)
        layout = QVBoxLayout()

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(freqs, amplitude, 'b-')
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('幅值')
        ax.set_title('单边幅值谱')
        ax.grid(True)

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        self.setLayout(layout)