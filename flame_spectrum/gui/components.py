# gui/components.py
from PyQt6.QtCore import QThread, pyqtSignal

class CalculationThread(QThread):
    """通用后台计算线程，适配多分子 HITRAN 引擎"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)          # 可选，当前未使用

    def __init__(self, hitran_engine, T, p, L, wn_start, wn_end, resolution, omega_wing):
        super().__init__()
        self.hitran = hitran_engine
        self.T = T
        self.p = p
        self.L = L
        self.wn_start = wn_start
        self.wn_end = wn_end
        self.resolution = resolution
        self.omega_wing = omega_wing

    def run(self):
        try:
            # 调用多分子引擎的 OD_mixture
            OD, Ab, Tr, wavenumber, total_coef, ind_coefs = self.hitran.OD_mixture(
                self.T, self.p, self.L,
                start=self.wn_start,
                end=self.wn_end,
                resolution=self.resolution,
                wing=self.omega_wing
            )
            results = {
                'wavenumber': wavenumber,
                'total_coef': total_coef,
                'individual_coefs': ind_coefs,
                'Tr': Tr,
                'Ab': Ab,
                'OD': OD,
                'params': {
                    'T': self.T,
                    'p': self.p,
                    'l': self.L,
                    'omega_wing': self.omega_wing
                }
            }
            self.finished.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")