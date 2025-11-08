# hitran_gui.py
import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QComboBox, QGroupBox, 
                             QFileDialog, QMessageBox, QTabWidget, QTextEdit, QCheckBox,
                             QDoubleSpinBox, QSpinBox, QProgressBar, QSplitter, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 强制使用思源黑体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Source Han Sans CN']  # 思源黑体
rcParams['axes.unicode_minus'] = False
print("使用思源黑体显示中文")
rcParams['text.usetex'] = False 
rcParams['mathtext.fontset'] = 'stix'

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# 导入HitranSpectrum类
from hitran_spectrum import HitranSpectrum

class CalculationThread(QThread):
    """计算线程"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, hitran, params):
        super().__init__()
        self.hitran = hitran
        self.params = params
    
    def run(self):
        try:
            # 执行计算
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

class MplCanvas(FigureCanvas):
    """Matplotlib画布"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

class HitranGUI(QMainWindow):
    """HITRAN光谱仿真GUI"""
    
    def __init__(self):
        super().__init__()
        self.hitran = None
        self.current_results = None
        self.calculation_thread = None
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle('HITRAN光谱仿真工具')
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # 右侧图形和结果显示
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 文件选择组
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout()
        
        # HITRAN文件选择
        hitran_layout = QHBoxLayout()
        self.par_file_edit = QLineEdit()
        self.par_file_edit.setPlaceholderText("选择HITRAN .par文件")
        par_file_btn = QPushButton("浏览")
        par_file_btn.clicked.connect(self.select_par_file)
        hitran_layout.addWidget(QLabel("HITRAN文件:"))
        hitran_layout.addWidget(self.par_file_edit)
        hitran_layout.addWidget(par_file_btn)
        file_layout.addLayout(hitran_layout)
        
        # 配分函数文件夹选择
        qfolder_layout = QHBoxLayout()
        self.q_folder_edit = QLineEdit()
        self.q_folder_edit.setPlaceholderText("选择配分函数文件夹 (包含q1.txt, q2.txt等)")
        q_folder_btn = QPushButton("浏览")
        q_folder_btn.clicked.connect(self.select_q_folder)
        qfolder_layout.addWidget(QLabel("配分函数文件夹:"))
        qfolder_layout.addWidget(self.q_folder_edit)
        qfolder_layout.addWidget(q_folder_btn)
        file_layout.addLayout(qfolder_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 分子信息显示组
        mol_info_group = QGroupBox("分子信息 (自动识别)")
        mol_info_layout = QVBoxLayout()
        
        # 分子名称
        mol_name_layout = QHBoxLayout()
        mol_name_layout.addWidget(QLabel("分子:"))
        self.molecule_label = QLabel("未识别")
        mol_name_layout.addWidget(self.molecule_label)
        mol_info_layout.addLayout(mol_name_layout)
        
        # 分子ID和同位素
        mol_id_layout = QHBoxLayout()
        mol_id_layout.addWidget(QLabel("分子ID:"))
        self.molecule_id_label = QLabel("--")
        mol_id_layout.addWidget(self.molecule_id_label)
        
        mol_id_layout.addWidget(QLabel("同位素ID:"))
        self.isotope_id_label = QLabel("--")
        mol_id_layout.addWidget(self.isotope_id_label)
        mol_info_layout.addLayout(mol_id_layout)
        
        # 分子质量
        mass_layout = QHBoxLayout()
        mass_layout.addWidget(QLabel("分子质量:"))
        self.mass_label = QLabel("未识别")
        mass_layout.addWidget(self.mass_label)
        mol_info_layout.addLayout(mass_layout)
        
        mol_info_group.setLayout(mol_info_layout)
        layout.addWidget(mol_info_group)
        
        # 计算参数组
        param_group = QGroupBox("计算参数")
        param_layout = QVBoxLayout()
        
        # 温度
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("温度 (K):"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(100, 5000)
        self.temp_spin.setValue(600)
        self.temp_spin.setSingleStep(10)
        temp_layout.addWidget(self.temp_spin)
        param_layout.addLayout(temp_layout)
        
        # 压力
        pressure_layout = QHBoxLayout()
        pressure_layout.addWidget(QLabel("压力 (atm):"))
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(0.001, 100)
        self.pressure_spin.setValue(1.0)
        self.pressure_spin.setSingleStep(0.1)
        pressure_layout.addWidget(self.pressure_spin)
        param_layout.addLayout(pressure_layout)
        
        # 浓度
        conc_layout = QHBoxLayout()
        conc_layout.addWidget(QLabel("浓度(ppm):"))
        self.conc_spin = QDoubleSpinBox()
        self.conc_spin.setRange(0.001, 1000000)
        self.conc_spin.setValue(10)
        self.conc_spin.setDecimals(3)
        self.conc_spin.setSingleStep(0.001)
        conc_layout.addWidget(self.conc_spin)
        param_layout.addLayout(conc_layout)
        
        # 光程长度
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("光程 (cm):"))
        self.path_spin = QDoubleSpinBox()
        self.path_spin.setRange(0.1, 10000)
        self.path_spin.setValue(100.0)
        self.path_spin.setSingleStep(10)
        path_layout.addWidget(self.path_spin)
        param_layout.addLayout(path_layout)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 波数范围组
        wavenumber_group = QGroupBox("波数范围")
        wavenumber_layout = QVBoxLayout()
        
        # 使用默认范围复选框
        self.use_default_range = QCheckBox("使用数据库默认范围")
        self.use_default_range.setChecked(True)
        self.use_default_range.toggled.connect(self.on_use_default_range_toggled)
        wavenumber_layout.addWidget(self.use_default_range)
        
        # 自定义范围
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("起始:"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 100000)
        self.start_spin.setValue(1870)
        self.start_spin.setDecimals(4)
        range_layout.addWidget(self.start_spin)
        
        range_layout.addWidget(QLabel("结束:"))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0, 100000)
        self.end_spin.setValue(2310)
        self.end_spin.setDecimals(4)
        range_layout.addWidget(self.end_spin)
        
        range_layout.addWidget(QLabel("分辨率:"))
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.0001, 1.0)
        self.resolution_spin.setValue(0.001)
        self.resolution_spin.setDecimals(4)
        range_layout.addWidget(self.resolution_spin)
        
        wavenumber_layout.addLayout(range_layout)
        
        # omega_wing参数
        wing_layout = QHBoxLayout()
        wing_layout.addWidget(QLabel("谱线计算域倍数:"))
        self.omega_wing_spin = QDoubleSpinBox()
        self.omega_wing_spin.setRange(1, 100)
        self.omega_wing_spin.setValue(10)
        self.omega_wing_spin.setSingleStep(1)
        wing_layout.addWidget(self.omega_wing_spin)
        wavenumber_layout.addLayout(wing_layout)
        
        wavenumber_group.setLayout(wavenumber_layout)
        layout.addWidget(wavenumber_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("加载数据")
        self.load_btn.clicked.connect(self.load_data)
        button_layout.addWidget(self.load_btn)
        
        self.calculate_btn = QPushButton("开始计算")
        self.calculate_btn.clicked.connect(self.calculate)
        self.calculate_btn.setEnabled(False)
        button_layout.addWidget(self.calculate_btn)
        
        # 添加清除图形按钮
        self.clear_btn = QPushButton("清除图形")
        self.clear_btn.clicked.connect(self.clear_plot)
        button_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("保存吸收系数")
        self.save_btn.clicked.connect(self.save_coefficients)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
        # 数据库信息
        info_group = QGroupBox("数据库信息")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """创建右侧面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 创建图形区域
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_widget.setLayout(plot_layout)
        
        # 创建图形画布
        self.canvas = MplCanvas(self, width=10, height=6, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        # 创建结果统计区域
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        stats_widget.setLayout(stats_layout)
        
        # 结果统计标题
        stats_title = QLabel("计算结果统计")
        stats_title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 5px;")
        stats_layout.addWidget(stats_title)
        
        # 结果统计文本
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        # 设置分割器
        splitter.addWidget(plot_widget)
        splitter.addWidget(stats_widget)
        splitter.setSizes([600, 200])  # 设置初始大小比例
        
        layout.addWidget(splitter)
        
        return panel

    def clear_plot(self):
        """清除图形"""
        self.canvas.axes.clear()
        
        # 重置坐标轴标签和标题
        self.canvas.axes.set_xlabel('波数 (cm$^{-1}$)')
        self.canvas.axes.set_ylabel('吸收系数 (cm$^{-1}$)')
        self.canvas.axes.set_title('')
        
        # 清除统计文本
        self.stats_text.clear()
        
        # 刷新图形
        self.canvas.draw()
        
        QMessageBox.information(self, "成功", "图形已清除")
    
    def on_use_default_range_toggled(self, checked):
        """使用默认范围复选框状态改变"""
        self.start_spin.setEnabled(not checked)
        self.end_spin.setEnabled(not checked)
    
    def select_par_file(self):
        """选择HITRAN文件"""
        filename, _ = QFileDialog.getOpenFileName(self, "选择HITRAN .par文件", "", "PAR Files (*.par)")
        if filename:
            self.par_file_edit.setText(filename)
    
    def select_q_folder(self):
        """选择配分函数文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择配分函数文件夹")
        if folder:
            self.q_folder_edit.setText(folder)
    
    def update_molecule_info(self, molecule_info):
        """更新分子信息显示"""
        if molecule_info:
            self.molecule_label.setText(molecule_info['molecule_name'])
            self.molecule_id_label.setText(str(molecule_info['molecule_id']))
            self.isotope_id_label.setText(str(molecule_info['isotope_id']))
            self.mass_label.setText(f"{molecule_info['molar_mass']:.6f} g/mol")
        else:
            self.molecule_label.setText("未识别")
            self.molecule_id_label.setText("--")
            self.isotope_id_label.setText("--")
            self.mass_label.setText("未识别")
    
    def load_data(self):
        """加载数据"""
        par_file = self.par_file_edit.text()
        q_folder = self.q_folder_edit.text()
        
        if not par_file or not q_folder:
            QMessageBox.warning(self, "错误", "请选择HITRAN文件和配分函数文件夹")
            return
        
        if not os.path.exists(par_file):
            QMessageBox.warning(self, "错误", "HITRAN文件不存在")
            return
        
        if not os.path.exists(q_folder):
            QMessageBox.warning(self, "错误", "配分函数文件夹不存在")
            return
        
        try:
            self.hitran = HitranSpectrum()
            self.hitran.load_data(par_file, q_folder)
            
            # 更新分子信息显示
            self.update_molecule_info(self.hitran.molecule_info)
            
            # 更新波数范围
            if self.use_default_range.isChecked():
                self.start_spin.setValue(self.hitran.default_start)
                self.end_spin.setValue(self.hitran.default_end)
            
            # 显示数据库信息
            self.info_text.setText(self.hitran.get_database_info())
            
            self.calculate_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "数据加载成功！")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载数据失败: {e}")
    
    def calculate(self):
        """开始计算"""
        if self.hitran is None:
            QMessageBox.warning(self, "错误", "请先加载数据")
            return
        
        try:
            # 禁用计算按钮，显示进度条
            self.calculate_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # 不确定进度
            
            # 获取计算参数
            T = self.temp_spin.value()
            p = self.pressure_spin.value()
            c = self.conc_spin.value()*1e-6
            l = self.path_spin.value()
            omega_wing = self.omega_wing_spin.value()
            
            # 确定波数范围
            if self.use_default_range.isChecked():
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
            
            # 在单独的线程中执行计算
            self.calculation_thread = CalculationThread(self.hitran, params)
            self.calculation_thread.finished.connect(self.on_calculation_finished)
            self.calculation_thread.error.connect(self.on_calculation_error)
            self.calculation_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {e}")
            self.calculate_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    def on_calculation_finished(self, results):
        """计算完成"""
        self.current_results = results
        
        # 更新图形
        self.update_plot()
        
        # 更新结果统计
        self.update_stats_text()
        
        # 恢复界面状态
        self.calculate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.save_btn.setEnabled(True)
        
        QMessageBox.information(self, "成功", "计算完成！")
    
    def on_calculation_error(self, error_msg):
        """计算错误"""
        QMessageBox.critical(self, "错误", f"计算失败: {error_msg}")
        self.calculate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def update_plot(self):
        """更新图形"""
        if self.current_results is None:
            return
        
        # 清除图形，确保每次都是重新绘制
        self.canvas.axes.clear()
        
        wavenumber = self.current_results['wavenumber']
        coef = self.current_results['coef']
        Tr = self.current_results['Tr']
        
        # 绘制吸收系数（左轴）
        ax1 = self.canvas.axes
        color = 'tab:blue'
        ax1.set_xlabel('波数 (cm$^{-1}$)')
        ax1.set_ylabel('吸收系数 (cm$^{-1}$)', color=color)
        line1 = ax1.plot(wavenumber, coef, color=color, linewidth=1, label='吸收系数')[0]
        ax1.tick_params(axis='y', labelcolor=color)
        
        # 创建右轴绘制透射率
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('透射率', color=color)
        line2 = ax2.plot(wavenumber, Tr, color=color, linestyle='-', linewidth=1, alpha=0.7, label='透射率')[0]
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1)  # 确保透射率范围正确
        
        # 添加图例
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        # 添加标题
        if self.hitran and self.hitran.molecule_info:
            mol_name = self.hitran.molecule_info['molecule_name']
            params = self.current_results['params']
            ax1.set_title(f'{mol_name} - 吸收系数和透射率\n'
                         f'T={params["T"]}K, p={params["p"]}atm, c={params["c"]:.2e}, l={params["l"]}cm')
        
        # 添加网格
        ax1.grid(True, alpha=0.3)
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def update_stats_text(self):
        """更新结果统计文本"""
        if self.current_results is None:
            return
        
        params = self.current_results['params']
        coef = self.current_results['coef']
        OD = self.current_results['OD']
        Tr = self.current_results['Tr']
        Ab = self.current_results['Ab']
        
        # 找到吸收系数最大的位置
        max_coef_idx = np.argmax(coef)
        max_coef_wavenumber = self.current_results['wavenumber'][max_coef_idx]
        
        # 找到透射率最小的位置
        min_Tr_idx = np.argmin(Tr)
        min_Tr_wavenumber = self.current_results['wavenumber'][min_Tr_idx]
        
        # 找到吸收率最大的位置
        max_Ab_idx = np.argmax(Ab)
        max_Ab_wavenumber = self.current_results['wavenumber'][max_Ab_idx]
        
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

数据点数: {len(self.current_results['wavenumber'])}
"""
        self.stats_text.setText(text)
    
    def save_coefficients(self):
        """保存吸收系数数据"""
        if self.current_results is None:
            QMessageBox.warning(self, "错误", "没有可保存的结果")
            return
        
        # 默认文件名
        default_filename = "absorption_coefficients.txt"
        if self.hitran and self.hitran.molecule_info:
            mol_name = self.hitran.molecule_info['molecule_name'].replace('(', '').replace(')', '')
            default_filename = f"{mol_name}_absorption_coefficients.txt"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存吸收系数数据", default_filename, "Text Files (*.txt)"
        )
        
        if filename:
            try:
                # 保存吸收系数数据（波数和吸收系数）
                data = np.column_stack((
                    self.current_results['wavenumber'],
                    self.current_results['coef']
                ))
                
                # 创建文件头
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
                
                QMessageBox.information(self, "成功", f"吸收系数数据已保存到: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

def main():
    app = QApplication(sys.argv)
    window = HitranGUI()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()
