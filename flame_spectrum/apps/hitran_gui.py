# apps/hitran_gui.py
# 
import sys
import os
# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QGroupBox, QFileDialog,
                             QMessageBox, QTextEdit, QCheckBox, QDoubleSpinBox,
                             QProgressBar, QSplitter)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Reset to Matplotlib's default, universally supported fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # Still keeps minus signs rendering correctly
plt.rcParams['text.usetex'] = True


from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 导入重构后的模块
from core.hitran_spectrum import HitranSpectrum
#from gui.components import MplCanvas, CalculationThread
from gui.components import SpectrumCanvas, CalculationThread

# Import the new reusable plotter
from gui.spectrum_plotter import SpectrumCanvas

class HitranGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.hitran = None
        self.current_results = None
        self.calculation_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('HITRAN 光谱仿真工具 (重构版)')
        self.setGeometry(100, 100, 1400, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)

        # 右侧图形和统计
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # ---- 文件选择 ----
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout()
        h_layout = QHBoxLayout()
        self.par_file_edit = QLineEdit()
        self.par_file_edit.setPlaceholderText("选择 .par 文件")
        btn_par = QPushButton("浏览")
        btn_par.clicked.connect(self.select_par_file)
        h_layout.addWidget(QLabel("HITRAN文件:"))
        h_layout.addWidget(self.par_file_edit)
        h_layout.addWidget(btn_par)
        file_layout.addLayout(h_layout)

        h_layout2 = QHBoxLayout()
        self.q_folder_edit = QLineEdit()
        self.q_folder_edit.setPlaceholderText("选择配分函数文件夹 (含 q*.txt)")
        btn_q = QPushButton("浏览")
        btn_q.clicked.connect(self.select_q_folder)
        h_layout2.addWidget(QLabel("Q文件夹:"))
        h_layout2.addWidget(self.q_folder_edit)
        h_layout2.addWidget(btn_q)
        file_layout.addLayout(h_layout2)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # ---- 分子信息 ----
        info_group = QGroupBox("分子信息 (自动识别)")
        info_layout = QVBoxLayout()
        h_mol = QHBoxLayout()
        h_mol.addWidget(QLabel("分子:"))
        self.molecule_label = QLabel("未识别")
        h_mol.addWidget(self.molecule_label)
        info_layout.addLayout(h_mol)

        h_id = QHBoxLayout()
        h_id.addWidget(QLabel("分子ID:"))
        self.molecule_id_label = QLabel("--")
        h_id.addWidget(self.molecule_id_label)
        h_id.addWidget(QLabel("同位素ID:"))
        self.isotope_id_label = QLabel("--")
        h_id.addWidget(self.isotope_id_label)
        info_layout.addLayout(h_id)

        h_mass = QHBoxLayout()
        h_mass.addWidget(QLabel("分子质量:"))
        self.mass_label = QLabel("未识别")
        h_mass.addWidget(self.mass_label)
        info_layout.addLayout(h_mass)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # ---- 计算参数 ----
        param_group = QGroupBox("计算参数")
        param_layout = QVBoxLayout()
        h_temp = QHBoxLayout()
        h_temp.addWidget(QLabel("温度 (K):"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(100, 5000)
        self.temp_spin.setValue(600)
        h_temp.addWidget(self.temp_spin)
        param_layout.addLayout(h_temp)

        h_press = QHBoxLayout()
        h_press.addWidget(QLabel("压力 (atm):"))
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(0.001, 100)
        self.pressure_spin.setValue(1.0)
        h_press.addWidget(self.pressure_spin)
        param_layout.addLayout(h_press)

        h_conc = QHBoxLayout()
        h_conc.addWidget(QLabel("浓度 (ppm):"))
        self.conc_spin = QDoubleSpinBox()
        self.conc_spin.setRange(0.001, 1e6)
        self.conc_spin.setValue(10)
        self.conc_spin.setDecimals(3)
        h_conc.addWidget(self.conc_spin)
        param_layout.addLayout(h_conc)

        h_path = QHBoxLayout()
        h_path.addWidget(QLabel("光程 (cm):"))
        self.path_spin = QDoubleSpinBox()
        self.path_spin.setRange(0.1, 10000)
        self.path_spin.setValue(100.0)
        h_path.addWidget(self.path_spin)
        param_layout.addLayout(h_path)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # ---- 波数范围 ----
        wn_group = QGroupBox("波数范围")
        wn_layout = QVBoxLayout()
        self.use_default_check = QCheckBox("使用数据库默认范围")
        self.use_default_check.setChecked(True)
        self.use_default_check.toggled.connect(self.on_use_default_toggled)
        wn_layout.addWidget(self.use_default_check)

        h_range = QHBoxLayout()
        h_range.addWidget(QLabel("起始:"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 100000)
        self.start_spin.setValue(1870)
        self.start_spin.setDecimals(4)
        h_range.addWidget(self.start_spin)
        h_range.addWidget(QLabel("结束:"))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0, 100000)
        self.end_spin.setValue(2310)
        self.end_spin.setDecimals(4)
        h_range.addWidget(self.end_spin)
        h_range.addWidget(QLabel("分辨率:"))
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.0001, 1.0)
        self.resolution_spin.setValue(0.001)
        self.resolution_spin.setDecimals(4)
        h_range.addWidget(self.resolution_spin)
        wn_layout.addLayout(h_range)

        h_wing = QHBoxLayout()
        h_wing.addWidget(QLabel("谱线计算域倍数:"))
        self.omega_wing_spin = QDoubleSpinBox()
        self.omega_wing_spin.setRange(1, 100)
        self.omega_wing_spin.setValue(10)
        h_wing.addWidget(self.omega_wing_spin)
        wn_layout.addLayout(h_wing)
        wn_group.setLayout(wn_layout)
        layout.addWidget(wn_group)

        # ---- 进度条 ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ---- 按钮 ----
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("加载数据")
        self.load_btn.clicked.connect(self.load_data)
        btn_layout.addWidget(self.load_btn)

        self.calc_btn = QPushButton("开始计算")
        self.calc_btn.clicked.connect(self.calculate)
        self.calc_btn.setEnabled(False)
        btn_layout.addWidget(self.calc_btn)

        self.clear_btn = QPushButton("清除图形")
        self.clear_btn.clicked.connect(self.clear_plot)
        btn_layout.addWidget(self.clear_btn)

        self.save_btn = QPushButton("保存吸收系数")
        self.save_btn.clicked.connect(self.save_coefficients)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)

        # ---- 数据库信息 ----
        info_db = QGroupBox("数据库信息")
        info_db_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        info_db_layout.addWidget(self.info_text)
        info_db.setLayout(info_db_layout)
        layout.addWidget(info_db)

        layout.addStretch()
        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        splitter = QSplitter(Qt.Orientation.Vertical)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_widget.setLayout(plot_layout)
        self.canvas = SpectrumCanvas(self, width=10, height=6, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        stats_widget.setLayout(stats_layout)
        stats_title = QLabel("计算结果统计")
        stats_title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 5px;")
        stats_layout.addWidget(stats_title)
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)

        splitter.addWidget(plot_widget)
        splitter.addWidget(stats_widget)
        splitter.setSizes([600, 200])
        layout.addWidget(splitter)
        return panel

    # ---------- 以下方法与原版相同，只调整了导入路径 ----------
    def select_par_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择 .par 文件", "", "PAR Files (*.par)")
        if filename:
            self.par_file_edit.setText(filename)

    def select_q_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择配分函数文件夹")
        if folder:
            self.q_folder_edit.setText(folder)

    def on_use_default_toggled(self, checked):
        self.start_spin.setEnabled(not checked)
        self.end_spin.setEnabled(not checked)

    def update_molecule_info(self, info):
        if info:
            self.molecule_label.setText(info['molecule_name'])
            self.molecule_id_label.setText(str(info['molecule_id']))
            self.isotope_id_label.setText(str(info['isotope_id']))
            self.mass_label.setText(f"{info['molar_mass']:.6f} g/mol")
        else:
            self.molecule_label.setText("未识别")
            self.molecule_id_label.setText("--")
            self.isotope_id_label.setText("--")
            self.mass_label.setText("未识别")

    def load_data(self):
        par_file = self.par_file_edit.text()
        q_folder = self.q_folder_edit.text()
        if not par_file or not q_folder:
            QMessageBox.warning(self, "错误", "请选择 .par 文件和配分函数文件夹")
            return
        if not os.path.exists(par_file):
            QMessageBox.warning(self, "错误", "HITRAN 文件不存在")
            return
        if not os.path.exists(q_folder):
            QMessageBox.warning(self, "错误", "配分函数文件夹不存在")
            return
        try:
            self.hitran = HitranSpectrum()
            self.hitran.load_data(par_file, q_folder)
            self.update_molecule_info(self.hitran.molecule_info)
            if self.use_default_check.isChecked():
                self.start_spin.setValue(self.hitran.default_start)
                self.end_spin.setValue(self.hitran.default_end)
            self.info_text.setText(self.hitran.get_database_info())
            self.calc_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "数据加载成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载数据失败: {e}")

    def calculate(self):
        if self.hitran is None:
            QMessageBox.warning(self, "错误", "请先加载数据")
            return
        try:
            self.calc_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            T = self.temp_spin.value()
            p = self.pressure_spin.value()
            c = self.conc_spin.value() * 1e-6
            l = self.path_spin.value()
            omega_wing = self.omega_wing_spin.value()
            if self.use_default_check.isChecked():
                start = None
                end = None
            else:
                start = self.start_spin.value()
                end = self.end_spin.value()
            resolution = self.resolution_spin.value()
            params = {
                'T': T, 'p': p, 'c': c, 'l': l,
                'start': start, 'end': end,
                'resolution': resolution, 'omega_wing': omega_wing
            }
            self.calculation_thread = CalculationThread(self.hitran, params)
            self.calculation_thread.finished.connect(self.on_calculation_finished)
            self.calculation_thread.error.connect(self.on_calculation_error)
            self.calculation_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {e}")
            self.calc_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def on_calculation_finished(self, results):
        self.current_results = results
        self.update_plot()
        self.update_stats_text()
        self.calc_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.save_btn.setEnabled(True)
        QMessageBox.information(self, "完成", "计算完成！")

    def on_calculation_error(self, error_msg):
        QMessageBox.critical(self, "错误", f"计算失败: {error_msg}")
        self.calc_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def clear_plot(self):
        self.canvas.axes.clear()
        self.canvas.axes.set_xlabel('波数 (cm$^{-1}$)')
        self.canvas.axes.set_ylabel('吸收系数 (cm$^{-1}$)')
        self.canvas.axes.set_title('')
        self.stats_text.clear()
        self.canvas.draw()
        QMessageBox.information(self, "成功", "图形已清除")

    def update_plot(self):
        if self.current_results is None:
            return

        wn = self.current_results['wavenumber']
        coef = self.current_results['coef']
        Tr = self.current_results['Tr']
        params = self.current_results['params']

        mol_name = self.hitran.molecule_info.get('molecule_name', 'HITRAN') if self.hitran else 'HITRAN'

        # 直接调用画线函数
        self.canvas.plot_spectrum(
            wavenumber=wn,
            coef=coef,
            transmittance=Tr,
            params=params,
            title=f"{mol_name} Absorption & Transmittance",
            show_abs=True,
            show_trans=True,
            grid_density='fine',
            show_wavelength=True
        )
    
    def update_stats_text(self):
        if self.current_results is None:
            return
        params = self.current_results['params']
        coef = self.current_results['coef']
        OD = self.current_results['OD']
        Tr = self.current_results['Tr']
        Ab = self.current_results['Ab']
        wavenumber = self.current_results['wavenumber']

        max_coef_idx = np.argmax(coef)
        max_coef_wavenumber = wavenumber[max_coef_idx]
        min_Tr_idx = np.argmin(Tr)
        min_Tr_wavenumber = wavenumber[min_Tr_idx]
        max_Ab_idx = np.argmax(Ab)
        max_Ab_wavenumber = wavenumber[max_Ab_idx]

        text = f"""计算参数:
温度: {params['T']} K    压力: {params['p']} atm    浓度: {params['c']:.2e}
光程: {params['l']} cm    分辨率: {params['resolution']} cm⁻¹

吸收系数:
最大值: {np.max(coef):.2e} cm⁻¹ (位于 {max_coef_wavenumber:.4f} cm⁻¹)
最小值: {np.min(coef):.2e} cm⁻¹
平均值: {np.mean(coef):.2e} cm⁻¹

透射率:
最小值: {np.min(Tr):.6f} (位于 {min_Tr_wavenumber:.4f} cm⁻¹)
最大值: {np.max(Tr):.6f}
平均值: {np.mean(Tr):.6f}

光学深度:
最大值: {np.max(OD):.6f}
最小值: {np.min(OD):.6f}

吸收率:
最大值: {np.max(Ab):.6f} (位于 {max_Ab_wavenumber:.4f} cm⁻¹)
最小值: {np.min(Ab):.6f}

数据点数: {len(wavenumber)}
"""
        self.stats_text.setText(text)

    def save_coefficients(self):
        if self.current_results is None:
            QMessageBox.warning(self, "错误", "没有可保存的结果")
            return
        default_name = "absorption_coefficients.txt"
        if self.hitran and self.hitran.molecule_info:
            mol_name = self.hitran.molecule_info['molecule_name'].replace('(', '').replace(')', '')
            default_name = f"{mol_name}_absorption_coefficients.txt"
        filename, _ = QFileDialog.getSaveFileName(self, "保存吸收系数数据", default_name, "Text Files (*.txt)")
        if filename:
            try:
                data = np.column_stack((self.current_results['wavenumber'], self.current_results['coef']))
                header = f"""# HITRAN吸收系数数据
# 分子: {self.hitran.molecule_info['molecule_name'] if self.hitran else 'Unknown'}
# 温度: {self.current_results['params']['T']} K
# 压力: {self.current_results['params']['p']} atm
# 浓度: {self.current_results['params']['c']:.2e}
# 光程: {self.current_results['params']['l']} cm
# 波数范围: {self.current_results['wavenumber'][0]:.4f} - {self.current_results['wavenumber'][-1]:.4f} cm⁻¹
# 分辨率: {self.current_results['params']['resolution']} cm⁻¹
#
# Wavenumber(cm-1)    Absorption_Coefficient(cm-1)
"""
                np.savetxt(filename, data, delimiter='\t', header=header, fmt='%.6e', comments='')
                QMessageBox.information(self, "成功", f"数据已保存到: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

def main():
    app = QApplication(sys.argv)
    window = HitranGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
