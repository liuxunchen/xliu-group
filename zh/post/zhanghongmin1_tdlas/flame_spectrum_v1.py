# combined_gas_spectrum_gui.py
import sys
import os
import numpy as np
import cantera as ct
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QTextEdit, QDoubleSpinBox,
                             QGroupBox, QMessageBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QSplitter, QComboBox, QTabWidget, QCheckBox,
                             QProgressBar, QFileDialog, QFrame, QScrollArea, QDialog)  # 添加QScrollArea和QDialog
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# 导入HitranSpectrum类（多分子版本）
sys.path.insert(0, 'voigt_simulation')
from hitran_spectrum_dual import HitranSpectrum

# 强制使用思源黑体
# 强制使用系统自带中文字体（Windows 微软雅黑）
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 替换为微软雅黑（Windows 自带）
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
rcParams['text.usetex'] = False
rcParams['mathtext.fontset'] = 'stix'
# 重复配置确保生效（可选，防止覆盖）
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


class GasCompositionSimulator:
    """气体组分模拟核心类"""

    def __init__(self):
        self.gas = None
        self.initial_state = None
        self.final_state = None
        self.results = None
        self.initial_conditions = {}
        self.mechanism_files = self.find_mechanism_files()

    def find_mechanism_files(self):
        """查找可用的反应机理文件"""
        mechanism_files = []
        possible_files = ['gri30.yaml', 'gri30.cti', 'gri3.0.yaml', 'gri3.0.cti']

        for file in possible_files:
            if os.path.exists(file):
                mechanism_files.append(file)

        # 也可以查找当前目录下的所有.yaml和.cti文件
        for file in os.listdir('.'):
            if file.endswith(('.yaml', '.cti')):
                if file not in mechanism_files:
                    mechanism_files.append(file)

        return mechanism_files

    def initialize_gas(self, mechanism_file='gri30.yaml'):
        """初始化气体对象"""
        try:
            self.gas = ct.Solution(mechanism_file)
            return True, f"成功加载反应机理: {mechanism_file}"
        except Exception as e:
            return False, f"加载反应机理失败: {e}"

    def get_thermodynamic_state(self, gas):
        """获取热力学状态（每摩尔）- 转换为kJ/mol单位，无小数"""
        return {
            'pressure': gas.P,
            'temperature': gas.T,
            'volume': gas.volume_mole,
            'internal_energy': int(gas.int_energy_mole / 1000),  # J/mol -> kJ/mol，取整
            'enthalpy': int(gas.enthalpy_mole / 1000),  # J/mol -> kJ/mol，取整
            'entropy': int(gas.entropy_mole / 1000),  # J/mol/K -> kJ/mol/K，取整
            'gibbs': int(gas.gibbs_mole / 1000),  # J/mol -> kJ/mol，取整
            'cp': int(gas.cp_mole / 1000),  # J/mol/K -> kJ/mol/K，取整
            'cv': int(gas.cv_mole / 1000),  # J/mol/K -> kJ/mol/K，取整
            'density': gas.density_mole,
            'mean_molecular_weight': gas.mean_molecular_weight
        }

    def calculate_equilibrium(self, T_initial, P_initial, composition, use_equivalence_ratio=False,
                              fuel=None, oxidizer=None, phi=1.0, equilibrate_method='HP'):
        """计算化学平衡"""
        try:
            # 设置初始状态
            if use_equivalence_ratio and fuel and oxidizer:
                self.gas.TP = T_initial, P_initial
                self.gas.set_equivalence_ratio(phi, fuel, oxidizer)
                self.initial_conditions.update({
                    'method': 'equivalence_ratio',
                    'fuel': fuel,
                    'oxidizer': oxidizer,
                    'phi': phi
                })
            else:
                self.gas.TPX = T_initial, P_initial, composition
                self.initial_conditions.update({
                    'method': 'direct_composition'
                })

            # 保存初始状态
            self.initial_state = self.get_thermodynamic_state(self.gas)
            initial_composition = dict(zip(self.gas.species_names, self.gas.X))

            # 平衡计算
            self.gas.equilibrate(equilibrate_method)

            # 保存最终状态
            self.final_state = self.get_thermodynamic_state(self.gas)
            final_composition = dict(zip(self.gas.species_names, self.gas.X))

            # 保存初始条件
            self.initial_conditions.update({
                'temperature': T_initial,
                'pressure': P_initial,
                'composition': composition,
                'equilibrate_method': equilibrate_method,
                'initial_composition': initial_composition
            })

            # 获取结果
            self.results = {
                'temperature': self.gas.T,
                'pressure': self.gas.P,
                'density': self.gas.density_mole,
                'mean_molecular_weight': self.gas.mean_molecular_weight,
                'mole_fractions': final_composition,
                'mass_fractions': dict(zip(self.gas.species_names, self.gas.Y)),
                'species_names': self.gas.species_names,
                'n_species': self.gas.n_species,
                'enthalpy': int(self.gas.enthalpy_mass / 1000),  # J/kg -> kJ/kg，取整
                'internal_energy': int(self.gas.int_energy_mass / 1000),  # J/kg -> kJ/kg，取整
                'entropy': int(self.gas.entropy_mass / 1000),  # J/kg/K -> kJ/kg/K，取整
                'cp': int(self.gas.cp_mass / 1000),  # J/kg/K -> kJ/kg/K，取整
                'cv': int(self.gas.cv_mass / 1000)  # J/kg/K -> kJ/kg/K，取整
            }

            return True, "平衡计算成功"

        except Exception as e:
            return False, f"平衡计算失败: {e}"

    def get_sorted_mole_fractions(self, threshold=1e-10):
        """获取按摩尔分数排序的结果"""
        if self.results is None:
            return None

        mole_fractions = self.results['mole_fractions']

        # 过滤掉太小的组分并按值排序
        filtered_fractions = {k: v for k, v in mole_fractions.items() if v >= threshold}
        sorted_fractions = dict(sorted(filtered_fractions.items(), key=lambda item: item[1], reverse=True))

        return sorted_fractions

    def get_thermodynamic_comparison_data(self):
        """获取热力学状态对比数据（用于表格显示）"""
        if self.initial_state is None or self.final_state is None:
            return None

        # 定义参数名称和单位
        parameters = [
            ('压力 (Pa)', 'pressure', '{:>12.0f}'),
            ('温度 (K)', 'temperature', '{:>12.0f}'),
            ('体积 (L/mol)', 'volume', '{:>12.6f}'),
            ('内能 (kJ/mol)', 'internal_energy', '{:>12.0f}'),
            ('焓 (kJ/mol)', 'enthalpy', '{:>12.0f}'),
            ('熵 (kJ/mol/K)', 'entropy', '{:>12.0f}'),
            ('吉布斯自由能 (kJ/mol)', 'gibbs', '{:>12.0f}'),
            ('定压比热 (kJ/mol/K)', 'cp', '{:>12.0f}'),
            ('定容比热 (kJ/mol/K)', 'cv', '{:>12.0f}'),
            ('密度 (kmol/m³)', 'density', '{:>12.6f}'),
            ('平均分子量 (kg/kmol)', 'mean_molecular_weight', '{:>12.6f}')
        ]

        comparison_data = []
        for param_name, param_key, format_str in parameters:
            initial_value = self.initial_state[param_key]
            final_value = self.final_state[param_key]

            comparison_data.append({
                'parameter': param_name,
                'initial': initial_value,
                'final': final_value,
                'initial_str': format_str.format(initial_value),
                'final_str': format_str.format(final_value)
            })

        return comparison_data

    def get_results_summary(self):
        """获取结果摘要"""
        if self.results is None:
            return "无计算结果"

        # 获取初始条件
        T_init = self.initial_conditions.get('temperature', 'N/A')
        P_init = self.initial_conditions.get('pressure', 'N/A')
        comp_method = self.initial_conditions.get('method', 'N/A')
        equil_method = self.initial_conditions.get('equilibrate_method', 'HP')

        if comp_method == 'equivalence_ratio':
            fuel = self.initial_conditions.get('fuel', 'N/A')
            oxidizer = self.initial_conditions.get('oxidizer', 'N/A')
            phi = self.initial_conditions.get('phi', 'N/A')
            comp_info = f"当量比方式: φ={phi}, 燃料={fuel}, 氧化剂={oxidizer}"
        else:
            composition = self.initial_conditions.get('composition', 'N/A')
            comp_info = f"直接组分: {composition}"

        summary = f"""=== 化学平衡计算结果 ===

初始条件:
温度: {T_init} K
压力: {P_init} Pa
平衡方法: {equil_method}
{comp_info}

主要组分 (摩尔分数 > 1e-6):
"""

        sorted_fractions = self.get_sorted_mole_fractions(threshold=1e-6)
        if sorted_fractions:
            for species, fraction in sorted_fractions.items():
                if fraction >= 1e-6:
                    summary += f"  {species:8} {fraction:12.6f} ({fraction * 100:8.4f}%)\n"
        else:
            summary += "  无满足条件的组分\n"

        return summary


class SpectrumCalculationThread(QThread):
    """光谱计算线程"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    info = pyqtSignal(str)  # 新增：用于发送信息消息

    def __init__(self, hitran, params):
        super().__init__()
        self.hitran = hitran
        self.params = params

    def run(self):
        try:
            # 重定向输出到GUI
            import io
            import contextlib

            # 创建字符串流来捕获输出
            output_stream = io.StringIO()

            # 重定向标准输出和标准错误
            with contextlib.redirect_stdout(output_stream), contextlib.redirect_stderr(output_stream):
                # 执行计算
                OD, Ab, Tr, wavenumber, total_coef, individual_ODs = self.hitran.OD_mixture(
                    self.params['T'], self.params['p'], self.params['l'],
                    start=self.params['start'], end=self.params['end'],
                    resolution=self.params['resolution'], omega_wing=self.params['omega_wing']
                )

            # 不再发送捕获的输出信息
            # if captured_output:
            #     for line in captured_output.split('\n'):
            #         if line.strip():
            #             self.info.emit(line.strip())

            results = {
                'wavenumber': wavenumber,
                'total_coef': total_coef,
                'OD': OD,
                'Ab': Ab,
                'Tr': Tr,
                'individual_ODs': individual_ODs,
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


class ThermodynamicTableWidget(QTableWidget):
    """热力学状态对比表格"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["参数", "初始状态", "平衡状态"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def update_data(self, comparison_data):
        """更新表格数据"""
        if comparison_data is None:
            self.setRowCount(0)
            return

        self.setRowCount(len(comparison_data))

        for row, data in enumerate(comparison_data):
            self.setItem(row, 0, QTableWidgetItem(data['parameter']))
            self.setItem(row, 1, QTableWidgetItem(data['initial_str']))
            self.setItem(row, 2, QTableWidgetItem(data['final_str']))


class MoleculeWidget(QWidget):
    """单个分子的文件选择和控制部件"""

    def __init__(self, parent=None, molecule_name="", folder_path=""):
        super().__init__(parent)
        self.molecule_name = molecule_name
        self.folder_path = folder_path
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # 分子名称标签
        name_label = QLabel(f"{self.molecule_name}文件:")
        name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(name_label)

        # 文件选择下拉框
        self.file_combo = QComboBox()
        self.update_file_list()
        layout.addWidget(self.file_combo)

        self.setLayout(layout)

    def update_file_list(self):
        """更新文件列表"""
        self.file_combo.clear()
        if os.path.exists(self.folder_path):
            for file in os.listdir(self.folder_path):
                if file.endswith('.par'):
                    full_path = os.path.join(self.folder_path, file)
                    self.file_combo.addItem(file, full_path)

        if self.file_combo.count() == 0:
            self.file_combo.addItem("未找到文件")


class CombinedGasSpectrumGUI(QMainWindow):
    """气体组分和光谱联合模拟GUI"""

    def __init__(self):
        super().__init__()
        self.gas_simulator = GasCompositionSimulator()
        self.spectrum_simulator = None
        self.current_spectrum_results = None
        self.calculation_thread = None

        # 文件路径设置
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.hitran_db_dir = os.path.join(self.base_dir, "hitran_database")
        self.q_folder_path = os.path.join(self.hitran_db_dir, "Q")

        # 定义可用的分子类型及其文件夹路径
        self.molecule_types = {
            "H2O": os.path.join(self.hitran_db_dir, "01_H2O"),
            "CO2": os.path.join(self.hitran_db_dir, "02_CO2"),
            "CO": os.path.join(self.hitran_db_dir, "05_CO"),
            "NO": os.path.join(self.hitran_db_dir, "08_NO"),
            "N2O": os.path.join(self.hitran_db_dir, "04_N2O"),
            "NO2": os.path.join(self.hitran_db_dir, "10_NO2")
        }

        # 当前选择的分子列表，初始包含H2O和CO2
        self.selected_molecules = ["H2O", "CO2"]

        # 存储分子部件的字典
        self.molecule_widgets = {}

        # 存储浓度控件的字典
        self.import_conc_combos = {}
        self.manual_conc_spins = {}

        self.init_ui()

    def find_par_files(self, folder_path):
        """查找指定文件夹中的par文件"""
        par_files = []
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.par'):
                    par_files.append(file)
        return sorted(par_files)

    def parse_par_file_range(self, par_file_path):
        """解析par文件的波数范围"""
        try:
            min_wavenumber = float('inf')
            max_wavenumber = float('-inf')
            line_count = 0

            with open(par_file_path, 'r') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        # HITRAN格式：波数在第3-15列
                        wavenumber_str = line[2:15].strip()
                        if wavenumber_str:
                            try:
                                wavenumber = float(wavenumber_str)
                                min_wavenumber = min(min_wavenumber, wavenumber)
                                max_wavenumber = max(max_wavenumber, wavenumber)
                                line_count += 1
                            except ValueError:
                                continue

            # 如果成功读取到数据，返回范围
            if line_count > 0 and min_wavenumber != float('inf') and max_wavenumber != float('-inf'):
                return min_wavenumber, max_wavenumber
            else:
                print(f"解析par文件失败: 未找到有效数据行，共读取 {line_count} 行")
                return None

        except Exception as e:
            print(f"解析par文件范围失败: {e}")
            return None

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle('气体化学平衡与光谱联合模拟 - 山东科技大学')
        self.setGeometry(100, 100, 1600, 1000)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 创建标签页
        self.tab_widget = QTabWidget()

        # 添加气体组分模拟标签页
        self.gas_tab = self.create_gas_tab()
        self.tab_widget.addTab(self.gas_tab, "气体化学平衡")

        # 添加光谱模拟标签页
        self.spectrum_tab = self.create_spectrum_tab()
        self.tab_widget.addTab(self.spectrum_tab, "光谱模拟")

        main_layout.addWidget(self.tab_widget)

    def create_gas_tab(self):
        """创建气体组分模拟标签页"""
        tab = QWidget()
        layout = QHBoxLayout()
        tab.setLayout(layout)

        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧控制面板
        control_panel = self.create_gas_control_panel()
        splitter.addWidget(control_panel)

        # 右侧结果显示
        result_panel = self.create_gas_result_panel()
        splitter.addWidget(result_panel)

        # 设置分割比例
        splitter.setSizes([400, 1000])

        layout.addWidget(splitter)

        return tab

    def create_gas_control_panel(self):
        """创建气体控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # 反应机理选择组
        mech_group = QGroupBox("反应机理设置")
        mech_layout = QVBoxLayout()

        mech_layout.addWidget(QLabel("选择反应机理文件:"))
        self.mech_combo = QComboBox()
        for mech_file in self.gas_simulator.mechanism_files:
            self.mech_combo.addItem(mech_file)
        if not self.gas_simulator.mechanism_files:
            self.mech_combo.addItem("gri30.yaml")
        mech_layout.addWidget(self.mech_combo)

        self.init_btn = QPushButton("初始化气体对象")
        self.init_btn.clicked.connect(self.initialize_gas)
        mech_layout.addWidget(self.init_btn)

        mech_group.setLayout(mech_layout)
        layout.addWidget(mech_group)

        # 初始条件组
        init_group = QGroupBox("初始条件")
        init_layout = QVBoxLayout()

        # 温度
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("初始温度 (K):"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(200, 5000)
        self.temp_spin.setValue(400.0)
        self.temp_spin.setSingleStep(10)
        temp_layout.addWidget(self.temp_spin)
        init_layout.addLayout(temp_layout)

        # 压力
        pressure_layout = QHBoxLayout()
        pressure_layout.addWidget(QLabel("初始压力 (Pa):"))
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(10, 10000000)
        self.pressure_spin.setValue(1000.0)
        self.pressure_spin.setSingleStep(1000)
        pressure_layout.addWidget(self.pressure_spin)
        init_layout.addLayout(pressure_layout)

        # 平衡方法
        equil_method_layout = QHBoxLayout()
        equil_method_layout.addWidget(QLabel("平衡方法:"))
        self.equil_method_combo = QComboBox()
        self.equil_method_combo.addItems(["HP", "TP", "SP", "SV", "TV", "UV"])
        equil_method_layout.addWidget(self.equil_method_combo)
        init_layout.addLayout(equil_method_layout)

        init_group.setLayout(init_layout)
        layout.addWidget(init_group)

        # 组分设置组
        comp_group = QGroupBox("组分设置")
        comp_layout = QVBoxLayout()

        # 输入方式选择
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("输入方式:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["直接组分", "当量比"])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        method_layout.addWidget(self.method_combo)
        comp_layout.addLayout(method_layout)

        # 直接组分输入
        self.direct_comp_widget = QWidget()
        direct_layout = QVBoxLayout()
        direct_layout.addWidget(QLabel("组分组成:"))
        self.comp_edit = QLineEdit()
        self.comp_edit.setText("CH4:0.5, O2:1, N2:3.76")
        self.comp_edit.setToolTip("格式: 分子式1:系数1, 分子式2:系数2, ...")
        direct_layout.addWidget(self.comp_edit)
        self.direct_comp_widget.setLayout(direct_layout)
        comp_layout.addWidget(self.direct_comp_widget)

        # 当量比输入
        self.equivalence_widget = QWidget()
        equivalence_layout = QVBoxLayout()

        fuel_layout = QHBoxLayout()
        fuel_layout.addWidget(QLabel("燃料:"))
        self.fuel_edit = QLineEdit()
        self.fuel_edit.setText("CH4:1")
        self.fuel_edit.setToolTip("格式: 分子式:系数")
        fuel_layout.addWidget(self.fuel_edit)
        equivalence_layout.addLayout(fuel_layout)

        oxidizer_layout = QHBoxLayout()
        oxidizer_layout.addWidget(QLabel("氧化剂:"))
        self.oxidizer_edit = QLineEdit()
        self.oxidizer_edit.setText("O2:1, N2:3.76")
        self.oxidizer_edit.setToolTip("格式: 分子式1:系数1, 分子式2:系数2, ...")
        oxidizer_layout.addWidget(self.oxidizer_edit)
        equivalence_layout.addLayout(oxidizer_layout)

        phi_layout = QHBoxLayout()
        phi_layout.addWidget(QLabel("当量比:"))
        self.phi_spin = QDoubleSpinBox()
        self.phi_spin.setRange(0.1, 5.0)
        self.phi_spin.setValue(1.0)
        self.phi_spin.setSingleStep(0.1)
        phi_layout.addWidget(self.phi_spin)
        equivalence_layout.addLayout(phi_layout)

        self.equivalence_widget.setLayout(equivalence_layout)
        comp_layout.addWidget(self.equivalence_widget)

        # 预设组分
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["自定义", "甲烷-空气", "乙烯-空气", "氢气-空气", "丙烷-空气"])
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        comp_layout.addLayout(preset_layout)

        comp_group.setLayout(comp_layout)
        layout.addWidget(comp_group)

        # 隐藏当量比部件（默认使用直接组分）
        self.equivalence_widget.setVisible(False)

        # 操作按钮
        button_layout = QVBoxLayout()

        self.calculate_btn = QPushButton("计算平衡")
        self.calculate_btn.clicked.connect(self.calculate_equilibrium)
        self.calculate_btn.setEnabled(False)
        button_layout.addWidget(self.calculate_btn)

        self.transfer_btn = QPushButton("传输到光谱模拟")
        self.transfer_btn.clicked.connect(self.transfer_to_spectrum)
        self.transfer_btn.setEnabled(False)
        button_layout.addWidget(self.transfer_btn)

        self.clear_btn = QPushButton("清除结果")
        self.clear_btn.clicked.connect(self.clear_gas_results)
        button_layout.addWidget(self.clear_btn)

        layout.addLayout(button_layout)

        layout.addStretch()

        return panel

    def create_gas_result_panel(self):
        """创建气体结果面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 热力学状态对比表格
        thermo_group = QGroupBox("热力学状态对比 (每摩尔)")
        thermo_layout = QVBoxLayout()

        # 使用自定义表格部件
        self.thermo_table = ThermodynamicTableWidget()
        thermo_layout.addWidget(self.thermo_table)

        thermo_group.setLayout(thermo_layout)
        splitter.addWidget(thermo_group)

        # 组分表格
        table_group = QGroupBox("主要组分 (摩尔分数 > 1e-6)")
        table_layout = QVBoxLayout()
        self.species_table = QTableWidget()
        self.species_table.setColumnCount(2)
        self.species_table.setHorizontalHeaderLabels(["组分", "摩尔分数"])
        self.species_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.species_table)
        table_group.setLayout(table_layout)
        splitter.addWidget(table_group)

        # 设置分割比例
        splitter.setSizes([500, 300])

        layout.addWidget(splitter)

        return panel

    def create_spectrum_tab(self):
        """创建光谱模拟标签页"""
        tab = QWidget()
        layout = QHBoxLayout()
        tab.setLayout(layout)

        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 创建可滚动的左侧控制面板
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        control_panel = self.create_spectrum_control_panel()
        scroll_area.setWidget(control_panel)
        splitter.addWidget(scroll_area)

        # 右侧图形和结果显示
        right_panel = self.create_spectrum_result_panel()
        splitter.addWidget(right_panel)

        # 设置分割比例
        splitter.setSizes([500, 1100])

        layout.addWidget(splitter)

        return tab

    def create_spectrum_control_panel(self):
        """创建光谱控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # 文件选择组
        file_group = QGroupBox("分子文件选择")
        file_layout = QVBoxLayout()

        # 配分函数文件夹选择
        qfolder_layout = QHBoxLayout()
        qfolder_layout.addWidget(QLabel("配分函数文件夹:"))
        self.q_folder_combo = QComboBox()
        if os.path.exists(self.q_folder_path):
            self.q_folder_combo.addItem(self.q_folder_path)
        else:
            self.q_folder_combo.addItem("请选择配分函数文件夹")
        qfolder_layout.addWidget(self.q_folder_combo)
        file_layout.addLayout(qfolder_layout)

        # 分子文件选择区域
        self.molecules_container = QWidget()
        self.molecules_layout = QVBoxLayout()
        self.molecules_container.setLayout(self.molecules_layout)

        # 创建初始的H2O和CO2分子部件
        for molecule in self.selected_molecules:
            if molecule in self.molecule_types:
                self.add_molecule_widget(molecule)

        file_layout.addWidget(self.molecules_container)

        # 添加/删除分子按钮
        buttons_layout = QHBoxLayout()

        self.add_molecule_btn = QPushButton("+ 添加分子")
        self.add_molecule_btn.clicked.connect(self.show_add_molecule_dialog)
        buttons_layout.addWidget(self.add_molecule_btn)

        self.remove_molecule_btn = QPushButton("- 删除分子")
        self.remove_molecule_btn.clicked.connect(self.remove_molecule)
        self.remove_molecule_btn.setEnabled(len(self.selected_molecules) > 1)
        buttons_layout.addWidget(self.remove_molecule_btn)

        buttons_layout.addStretch()
        file_layout.addLayout(buttons_layout)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 从气体平衡导入组
        self.import_group = self.create_import_group()
        layout.addWidget(self.import_group)

        # 手动参数组
        self.manual_group = self.create_manual_group()
        layout.addWidget(self.manual_group)

        # 光程长度
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("光程 (cm):"))
        self.path_spin = QDoubleSpinBox()
        self.path_spin.setRange(0.1, 10000)
        self.path_spin.setValue(10.0)
        self.path_spin.setSingleStep(10)
        path_layout.addWidget(self.path_spin)
        layout.addLayout(path_layout)

        # 波数范围组
        wavenumber_group = QGroupBox("波数范围")
        wavenumber_layout = QVBoxLayout()

        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("起始:"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 100000)
        self.start_spin.setValue(4650)
        self.start_spin.setDecimals(4)
        range_layout.addWidget(self.start_spin)

        range_layout.addWidget(QLabel("结束:"))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0, 100000)
        self.end_spin.setValue(5410)
        self.end_spin.setDecimals(4)
        range_layout.addWidget(self.end_spin)

        range_layout.addWidget(QLabel("分辨率:"))
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.0001, 1.0)
        self.resolution_spin.setValue(0.01)
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

        self.load_spectrum_btn = QPushButton("加载分子数据")
        self.load_spectrum_btn.clicked.connect(self.load_spectrum_data)
        button_layout.addWidget(self.load_spectrum_btn)

        self.calculate_spectrum_btn = QPushButton("开始计算")
        self.calculate_spectrum_btn.clicked.connect(self.calculate_spectrum)
        self.calculate_spectrum_btn.setEnabled(False)
        button_layout.addWidget(self.calculate_spectrum_btn)

        self.clear_spectrum_btn = QPushButton("清除光谱")
        self.clear_spectrum_btn.clicked.connect(self.clear_spectrum_results)
        self.clear_spectrum_btn.setEnabled(False)
        button_layout.addWidget(self.clear_spectrum_btn)

        layout.addLayout(button_layout)

        # 分子信息
        info_group = QGroupBox("分子信息")
        info_layout = QVBoxLayout()
        self.spectrum_info_text = QTextEdit()
        self.spectrum_info_text.setMaximumHeight(150)
        self.spectrum_info_text.setReadOnly(True)
        info_layout.addWidget(self.spectrum_info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()

        return panel
    def create_spectrum_result_panel(self):
        """创建光谱结果面板"""
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
        self.stats_text.setMaximumHeight(300)  # 增加高度以容纳更多信息
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)

        # 设置分割器（只显示图形和统计结果）
        splitter.addWidget(plot_widget)
        splitter.addWidget(stats_widget)
        splitter.setSizes([600, 400])  # 设置初始大小比例

        layout.addWidget(splitter)

        return panel

    def create_import_group(self):
        """创建从气体平衡导入组"""
        import_group = QGroupBox("从气体平衡导入")
        import_layout = QVBoxLayout()

        self.import_gas_check = QCheckBox("使用气体平衡结果")
        self.import_gas_check.toggled.connect(self.on_import_gas_toggled)
        import_layout.addWidget(self.import_gas_check)

        # 温度显示
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("温度 (K):"))
        self.import_temp_label = QLabel("400.0")
        temp_layout.addWidget(self.import_temp_label)
        import_layout.addLayout(temp_layout)

        # 压力显示
        pressure_layout = QHBoxLayout()
        pressure_layout.addWidget(QLabel("压力 (atm):"))
        self.import_pressure_label = QLabel("--")
        pressure_layout.addWidget(self.import_pressure_label)
        import_layout.addLayout(pressure_layout)

        # 存储浓度下拉框的字典
        self.import_conc_combos = {}

        # 初始为选中的分子创建浓度下拉框
        for molecule in self.selected_molecules:
            self.add_import_concentration_combo(molecule, import_layout)

        import_group.setLayout(import_layout)
        return import_group

    def create_manual_group(self):
        """创建手动参数设置组"""
        manual_group = QGroupBox("手动参数设置")
        self.manual_layout = QVBoxLayout()

        # 温度
        manual_temp_layout = QHBoxLayout()
        manual_temp_layout.addWidget(QLabel("温度 (K):"))
        self.manual_temp_spin = QDoubleSpinBox()
        self.manual_temp_spin.setRange(100, 5000)
        self.manual_temp_spin.setValue(600)
        self.manual_temp_spin.setSingleStep(10)
        manual_temp_layout.addWidget(self.manual_temp_spin)
        self.manual_layout.addLayout(manual_temp_layout)

        # 压力
        manual_pressure_layout = QHBoxLayout()
        manual_pressure_layout.addWidget(QLabel("压力 (atm):"))
        self.manual_pressure_spin = QDoubleSpinBox()
        self.manual_pressure_spin.setRange(0.0001, 10)
        self.manual_pressure_spin.setValue(0.001)
        self.manual_pressure_spin.setSingleStep(0.001)
        self.manual_pressure_spin.setDecimals(4)
        manual_pressure_layout.addWidget(self.manual_pressure_spin)
        self.manual_layout.addLayout(manual_pressure_layout)

        # 存储手动浓度控件的字典
        self.manual_conc_spins = {}

        # 初始为选中的分子创建浓度输入框
        for molecule in self.selected_molecules:
            self.add_manual_concentration_spin(molecule, self.manual_layout)

        manual_group.setLayout(self.manual_layout)
        return manual_group

    def add_molecule_widget(self, molecule_name):
        """添加分子文件选择部件"""
        if molecule_name in self.molecule_types and molecule_name not in self.molecule_widgets:
            folder_path = self.molecule_types[molecule_name]
            widget = MoleculeWidget(self, molecule_name, folder_path)
            self.molecule_widgets[molecule_name] = widget
            self.molecules_layout.addWidget(widget)

            # 连接文件变化信号
            widget.file_combo.currentTextChanged.connect(
                lambda text, mol=molecule_name: self.on_molecule_file_changed(mol, text)
            )

    def remove_molecule_widget(self, molecule_name):
        """移除分子文件选择部件"""
        if molecule_name in self.molecule_widgets:
            widget = self.molecule_widgets.pop(molecule_name)
            widget.deleteLater()
            self.molecules_layout.removeWidget(widget)

    def add_import_concentration_combo(self, molecule_name, parent_layout):
        """为导入组添加浓度下拉框"""
        if molecule_name not in self.import_conc_combos:
            conc_layout = QHBoxLayout()
            conc_layout.addWidget(QLabel(f"{molecule_name}浓度:"))
            combo = QComboBox()
            combo.addItem("未找到", 0.0)
            self.import_conc_combos[molecule_name] = combo
            conc_layout.addWidget(combo)
            parent_layout.addLayout(conc_layout)

    def remove_import_concentration_combo(self, molecule_name):
        """从导入组移除浓度下拉框"""
        if molecule_name in self.import_conc_combos:
            combo = self.import_conc_combos.pop(molecule_name)
            # 找到并移除对应的布局
            for i in range(self.import_group.layout().count()):
                item = self.import_group.layout().itemAt(i)
                if item and item.layout():
                    for j in range(item.layout().count()):
                        sub_item = item.layout().itemAt(j)
                        if sub_item and sub_item.widget() == combo:
                            # 移除整个布局
                            layout_to_remove = item.layout()
                            # 清理布局中的部件
                            while layout_to_remove.count():
                                child = layout_to_remove.takeAt(0)
                                if child.widget():
                                    child.widget().deleteLater()
                            break

    def add_manual_concentration_spin(self, molecule_name, parent_layout):
        """为手动组添加浓度输入框"""
        if molecule_name not in self.manual_conc_spins:
            conc_layout = QHBoxLayout()
            conc_layout.addWidget(QLabel(f"{molecule_name}浓度:"))
            spin = QDoubleSpinBox()
            spin.setRange(0, 1.0)
            spin.setValue(0.1)
            spin.setDecimals(6)
            self.manual_conc_spins[molecule_name] = spin
            conc_layout.addWidget(spin)
            parent_layout.addLayout(conc_layout)

    def remove_manual_concentration_spin(self, molecule_name):
        """从手动组移除浓度输入框"""
        if molecule_name in self.manual_conc_spins:
            spin = self.manual_conc_spins.pop(molecule_name)
            # 找到并移除对应的布局
            for i in range(self.manual_layout.count()):
                item = self.manual_layout.itemAt(i)
                if item and item.layout():
                    for j in range(item.layout().count()):
                        sub_item = item.layout().itemAt(j)
                        if sub_item and sub_item.widget() == spin:
                            # 移除整个布局
                            layout_to_remove = item.layout()
                            # 清理布局中的部件
                            while layout_to_remove.count():
                                child = layout_to_remove.takeAt(0)
                                if child.widget():
                                    child.widget().deleteLater()
                            break

    def show_add_molecule_dialog(self):
        """显示添加分子对话框"""
        # 获取尚未选择的分子类型
        available_molecules = [mol for mol in self.molecule_types.keys()
                               if mol not in self.selected_molecules]

        if not available_molecules:
            QMessageBox.information(self, "提示", "所有可用分子都已添加")
            return

        # 创建简单的选择对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择要添加的分子")
        layout = QVBoxLayout()

        label = QLabel("选择要添加的分子类型:")
        layout.addWidget(label)

        combo = QComboBox()
        combo.addItems(available_molecules)
        layout.addWidget(combo)

        button_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")

        ok_btn.clicked.connect(lambda: self.add_molecule(combo.currentText(), dialog))
        cancel_btn.clicked.connect(dialog.reject)

        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.exec()

    def add_molecule(self, molecule_name, dialog=None):
        """添加分子"""
        if molecule_name in self.molecule_types and molecule_name not in self.selected_molecules:
            self.selected_molecules.append(molecule_name)

            # 添加文件选择部件
            self.add_molecule_widget(molecule_name)

            # 为导入组添加浓度下拉框
            self.add_import_concentration_combo(molecule_name, self.import_group.layout())

            # 为手动组添加浓度输入框
            self.add_manual_concentration_spin(molecule_name, self.manual_layout)

            # 启用删除按钮
            self.remove_molecule_btn.setEnabled(len(self.selected_molecules) > 1)

            if dialog:
                dialog.accept()

            # 更新信息显示
            self.spectrum_info_text.append(f"已添加分子: {molecule_name}")

    def remove_molecule(self):
        """删除分子"""
        if len(self.selected_molecules) <= 1:
            QMessageBox.warning(self, "警告", "至少需要保留一个分子")
            return

        # 创建选择对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择要删除的分子")
        layout = QVBoxLayout()

        label = QLabel("选择要删除的分子类型:")
        layout.addWidget(label)

        combo = QComboBox()
        combo.addItems(self.selected_molecules)
        layout.addWidget(combo)

        button_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")

        ok_btn.clicked.connect(lambda: self.do_remove_molecule(combo.currentText(), dialog))
        cancel_btn.clicked.connect(dialog.reject)

        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.exec()

    def do_remove_molecule(self, molecule_name, dialog=None):
        """执行删除分子"""
        if molecule_name in self.selected_molecules and len(self.selected_molecules) > 1:
            self.selected_molecules.remove(molecule_name)

            # 移除文件选择部件
            self.remove_molecule_widget(molecule_name)

            # 从导入组移除浓度下拉框
            self.remove_import_concentration_combo(molecule_name)

            # 从手动组移除浓度输入框
            self.remove_manual_concentration_spin(molecule_name)

            # 更新删除按钮状态
            self.remove_molecule_btn.setEnabled(len(self.selected_molecules) > 1)

            if dialog:
                dialog.accept()

            # 更新信息显示
            self.spectrum_info_text.append(f"已删除分子: {molecule_name}")

    def on_molecule_file_changed(self, molecule_name, text):
        """分子文件选择变化"""
        if text and text != "未找到文件":
            self.spectrum_info_text.append(f"{molecule_name}文件已选择: {text}")

    def auto_set_wavenumber_range(self):
        """自动设置波数范围"""
        min_start = float('inf')
        max_end = float('-inf')
        files_processed = 0

        # 检查所有选中的分子文件
        for molecule_name in self.selected_molecules:
            if molecule_name in self.molecule_widgets:
                widget = self.molecule_widgets[molecule_name]
                file_path = widget.file_combo.currentData()

                if file_path and os.path.exists(file_path):
                    file_range = self.parse_par_file_range(file_path)
                    if file_range:
                        start, end = file_range
                        min_start = min(min_start, start)
                        max_end = max(max_end, end)
                        files_processed += 1
                        self.spectrum_info_text.append(
                            f"{molecule_name}文件范围: {start:.2f} - {end:.2f} cm⁻¹"
                        )

        # 设置波数范围
        if files_processed > 0 and min_start != float('inf') and max_end != float('-inf'):
            if min_start < max_end:
                range_extension = (max_end - min_start) * 0.05
                start_range = max(0, min_start - range_extension)
                end_range = max_end + range_extension

                self.start_spin.setValue(round(start_range, 2))
                self.end_spin.setValue(round(end_range, 2))

                self.spectrum_info_text.append(
                    f"自动设置波数范围: {round(start_range, 2)} - {round(end_range, 2)} cm⁻¹"
                )
                return True

        self.spectrum_info_text.append("警告：无法自动设置波数范围，请手动设置")
        return False
    # 气体模拟相关方法
    def on_method_changed(self, method):
        """输入方式改变"""
        if method == "直接组分":
            self.direct_comp_widget.setVisible(True)
            self.equivalence_widget.setVisible(False)
        else:
            self.direct_comp_widget.setVisible(False)
            self.equivalence_widget.setVisible(True)

    def on_preset_changed(self, preset_name):
        """预设组分改变"""
        presets = {
            "甲烷-空气": ("CH4:0.5, O2:1, N2:3.76", "CH4:1", "O2:1, N2:3.76"),
            "乙烯-空气": ("C2H4:0.3, O2:1, N2:3.76", "C2H4:1", "O2:1, N2:3.76"),
            "氢气-空气": ("H2:1, O2:0.5, N2:1.88", "H2:1", "O2:0.5, N2:1.88"),
            "丙烷-空气": ("C3H8:0.2, O2:1, N2:3.76", "C3H8:1", "O2:1, N2:3.76")
        }

        if preset_name in presets:
            direct_comp, fuel, oxidizer = presets[preset_name]
            self.comp_edit.setText(direct_comp)
            self.fuel_edit.setText(fuel)
            self.oxidizer_edit.setText(oxidizer)

    def initialize_gas(self):
        """初始化气体对象"""
        mechanism_file = self.mech_combo.currentText()
        success, message = self.gas_simulator.initialize_gas(mechanism_file)

        if success:
            QMessageBox.information(self, "成功", message)
            self.calculate_btn.setEnabled(True)

            # 显示可用的物种信息
            if self.gas_simulator.gas:
                species_info = f"反应机理加载成功！\n"
                species_info += f"物种数量: {self.gas_simulator.gas.n_species}\n"
                species_info += f"反应数量: {self.gas_simulator.gas.n_reactions}\n"
                species_info += f"前10个物种: {', '.join(self.gas_simulator.gas.species_names[:10])}..."
                # 清除之前的表格数据
                self.thermo_table.setRowCount(0)
        else:
            QMessageBox.critical(self, "错误", message)

    def calculate_equilibrium(self):
        """计算平衡"""
        try:
            T_initial = self.temp_spin.value()
            P_initial = self.pressure_spin.value()
            equil_method = self.equil_method_combo.currentText()

            if self.method_combo.currentText() == "直接组分":
                composition = self.comp_edit.text()
                success, message = self.gas_simulator.calculate_equilibrium(
                    T_initial, P_initial, composition, equilibrate_method=equil_method
                )
            else:
                fuel = self.fuel_edit.text()
                oxidizer = self.oxidizer_edit.text()
                phi = self.phi_spin.value()
                success, message = self.gas_simulator.calculate_equilibrium(
                    T_initial, P_initial, "", True, fuel, oxidizer, phi, equil_method
                )

            if success:
                # 更新热力学状态对比表格
                comparison_data = self.gas_simulator.get_thermodynamic_comparison_data()
                self.thermo_table.update_data(comparison_data)

                # 更新组分表格
                self.update_species_table()

                # 启用传输按钮
                self.transfer_btn.setEnabled(True)

                QMessageBox.information(self, "成功", message)
            else:
                QMessageBox.critical(self, "错误", message)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {e}")

    def update_species_table(self):
        """更新组分表格"""
        sorted_fractions = self.gas_simulator.get_sorted_mole_fractions(threshold=1e-6)

        if sorted_fractions:
            self.species_table.setRowCount(len(sorted_fractions))

            for row, (species, fraction) in enumerate(sorted_fractions.items()):
                self.species_table.setItem(row, 0, QTableWidgetItem(species))
                if fraction >= 1e-6:
                    fraction_str = f"{fraction:.6f}"
                else:
                    fraction_str = f"{fraction:.2e}"
                self.species_table.setItem(row, 1, QTableWidgetItem(fraction_str))

    def clear_gas_results(self):
        """清除气体结果"""
        self.thermo_table.setRowCount(0)
        self.species_table.setRowCount(0)
        self.gas_simulator.results = None
        self.gas_simulator.initial_state = None
        self.gas_simulator.final_state = None
        self.gas_simulator.initial_conditions = {}
        self.transfer_btn.setEnabled(False)
        QMessageBox.information(self, "成功", "结果已清除")

    def transfer_to_spectrum(self):
        """将气体平衡结果传输到光谱模拟"""
        if self.gas_simulator.results is None:
            QMessageBox.warning(self, "警告", "没有可传输的计算结果")
            return

        # 更新温度
        temperature = self.gas_simulator.results['temperature']
        self.import_temp_label.setText(f"{temperature:.2f}")
        self.manual_temp_spin.setValue(temperature)

        # 更新压力（转换为atm）
        pressure_pa = self.gas_simulator.results['pressure']
        pressure_atm = pressure_pa / 101325.0
        self.import_pressure_label.setText(f"{pressure_atm:.4f}")
        self.manual_pressure_spin.setValue(pressure_atm)

        # 更新浓度选择框
        mole_fractions = self.gas_simulator.results['mole_fractions']

        # 更新所有选中分子的浓度下拉框
        for molecule_name in self.selected_molecules:
            if molecule_name in self.import_conc_combos:
                combo = self.import_conc_combos[molecule_name]
                combo.clear()

                # 查找该分子的所有相关物种
                options = []
                for species, fraction in mole_fractions.items():
                    # 精确匹配分子名称
                    if molecule_name.upper() == species.upper():
                        options.append((species, fraction))
                    # 或者包含分子名称（如H2O可能包含在H2O(L)中）
                    elif molecule_name.upper() in species.upper():
                        options.append((species, fraction))

                if options:
                    for species, fraction in options:
                        combo.addItem(f"{species}: {fraction:.6f}", fraction)
                    combo.setCurrentIndex(0)
                else:
                    combo.addItem(f"未找到{molecule_name}", 0.0)

        # 切换到光谱模拟标签页
        self.tab_widget.setCurrentIndex(1)
        QMessageBox.information(self, "成功", "气体平衡结果已传输到光谱模拟")

    # 光谱模拟相关方法
    def on_import_gas_toggled(self, checked):
        """导入气体平衡结果复选框状态改变"""
        self.manual_temp_spin.setEnabled(not checked)
        self.manual_pressure_spin.setEnabled(not checked)
        self.manual_h2o_spin.setEnabled(not checked)
        self.manual_co2_spin.setEnabled(not checked)
        self.manual_co_spin.setEnabled(not checked)
        self.manual_no_spin.setEnabled(not checked)
        self.manual_no2_spin.setEnabled(not checked)
    def load_spectrum_data(self):
        """加载光谱数据"""
        q_folder = self.q_folder_combo.currentText()

        # 验证配分函数文件夹
        if not q_folder or not os.path.exists(q_folder):
            QMessageBox.warning(self, "错误", "请选择有效的配分函数文件夹")
            return

        # 验证所有选中的分子文件
        missing_files = []
        molecule_files = {}

        for molecule_name in self.selected_molecules:
            if molecule_name in self.molecule_widgets:
                widget = self.molecule_widgets[molecule_name]
                file_path = widget.file_combo.currentData()

                if file_path and os.path.exists(file_path):
                    molecule_files[molecule_name] = file_path
                else:
                    missing_files.append(molecule_name)

        if missing_files:
            QMessageBox.warning(self, "错误", f"以下分子文件不存在: {', '.join(missing_files)}")
            return

        try:
            self.spectrum_simulator = HitranSpectrum(q_folder=q_folder)

            # 获取浓度并添加分子
            for molecule_name in self.selected_molecules:
                file_path = molecule_files[molecule_name]

                if self.import_gas_check.isChecked():
                    # 从导入的浓度下拉框获取浓度
                    if molecule_name in self.import_conc_combos:
                        conc = self.import_conc_combos[molecule_name].currentData()
                    else:
                        conc = 0.0
                else:
                    # 从手动输入框获取浓度
                    if molecule_name in self.manual_conc_spins:
                        conc = self.manual_conc_spins[molecule_name].value()
                    else:
                        conc = 0.1

                # 添加分子到光谱模拟器
                self.spectrum_simulator.add_molecule(file_path, concentration=conc,
                                                     molecule_name=molecule_name)

            # 显示分子信息
            self.spectrum_info_text.setText(self.spectrum_simulator.get_molecule_info())

            # 自动设置波数范围
            self.spectrum_info_text.append("正在自动设置波数范围...")
            success = self.auto_set_wavenumber_range()
            if success:
                self.spectrum_info_text.append("波数范围自动设置成功！")
            else:
                self.spectrum_info_text.append("警告：无法自动设置波数范围，请手动设置")

            self.calculate_spectrum_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "分子数据加载成功！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载分子数据失败: {e}")

    def calculate_spectrum(self):
        """开始光谱计算"""
        if self.spectrum_simulator is None:
            QMessageBox.warning(self, "错误", "请先加载分子数据")
            return

        try:
            # 禁用计算按钮，显示进度条
            self.calculate_spectrum_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # 不确定进度

            # 获取计算参数
            if self.import_gas_check.isChecked():
                T = float(self.import_temp_label.text())
                p = float(self.import_pressure_label.text())
            else:
                T = self.manual_temp_spin.value()
                p = self.manual_pressure_spin.value()

            l = self.path_spin.value()
            start = self.start_spin.value()
            end = self.end_spin.value()
            resolution = self.resolution_spin.value()
            omega_wing = self.omega_wing_spin.value()

            params = {
                'T': T, 'p': p, 'l': l,
                'start': start, 'end': end,
                'resolution': resolution, 'omega_wing': omega_wing
            }

            # 在单独的线程中执行计算
            self.calculation_thread = SpectrumCalculationThread(self.spectrum_simulator, params)
            self.calculation_thread.finished.connect(self.on_spectrum_calculation_finished)
            self.calculation_thread.error.connect(self.on_spectrum_calculation_error)
            # 不再连接info信号，不显示程序输出信息
            self.calculation_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {e}")
            self.calculate_spectrum_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def on_spectrum_calculation_finished(self, results):
        """光谱计算完成"""
        self.current_spectrum_results = results

        # 更新图形
        self.update_spectrum_plot()

        # 更新结果统计
        self.update_spectrum_stats_text()

        # 恢复界面状态
        self.calculate_spectrum_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.clear_spectrum_btn.setEnabled(True)

        QMessageBox.information(self, "成功", "光谱计算完成！")

    def on_spectrum_calculation_error(self, error_msg):
        """光谱计算错误"""
        QMessageBox.critical(self, "错误", f"光谱计算失败: {error_msg}")
        self.calculate_spectrum_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def clear_spectrum_results(self):
        """清除光谱结果"""
        # 清除图形
        self.canvas.axes.clear()

        # 重置坐标轴范围到默认值
        self.canvas.axes.set_xlim(auto=True)
        self.canvas.axes.set_ylim(auto=True)

        # 重新设置坐标轴标签和标题
        self.canvas.axes.set_xlabel('波数 (cm$^{-1}$)')
        self.canvas.axes.set_ylabel('吸收系数 (cm$^{-1}$)')
        self.canvas.axes.set_title('混合气体吸收光谱')
        self.canvas.axes.grid(True, alpha=0.3)

        # 如果存在右轴，也清除并重置
        for ax in self.canvas.fig.get_axes():
            if ax != self.canvas.axes:
                ax.remove()

        self.canvas.draw()

        # 清除结果统计
        self.stats_text.clear()

        # 清除当前结果
        self.current_spectrum_results = None

        # 禁用清除按钮
        self.clear_spectrum_btn.setEnabled(False)

        QMessageBox.information(self, "成功", "光谱结果已清除")

    def update_spectrum_plot(self):
        """更新光谱图形 - 显示H2O、CO2、NO的吸收系数，以及叠加的透射率光谱"""
        if self.current_spectrum_results is None:
            return

        # 清除图形
        self.canvas.axes.clear()

        wavenumber = self.current_spectrum_results['wavenumber']
        individual_ODs = self.current_spectrum_results['individual_ODs']
        Tr = self.current_spectrum_results['Tr']

        # 计算各分子的吸收系数
        individual_coefs = {}
        for molecule_name, od in individual_ODs.items():
            coef_single = od / (self.current_spectrum_results['params']['p'] * 101325 /
                              (1.380649e-23 * self.current_spectrum_results['params']['T']) *
                              self.current_spectrum_results['params']['l'])
            individual_coefs[molecule_name] = coef_single

        # 绘制各分子的吸收系数（分别显示）
        colors = {'H2O': 'tab:blue', 'CO2': 'tab:green', 'NO': 'tab:purple', 'CO': 'tab:red','N2O': 'tab:pink', 'NO2': 'tab:brown'}
        lines = []

        # 绘制H2O吸收系数
        if 'H2O' in individual_coefs:
            line_h2o = self.canvas.axes.plot(wavenumber, individual_coefs['H2O'],
                                             color=colors['H2O'], linewidth=1, label='H2O吸收系数')[0]
            lines.append(line_h2o)

        # 绘制CO2吸收系数
        if 'CO2' in individual_coefs:
            line_co2 = self.canvas.axes.plot(wavenumber, individual_coefs['CO2'],
                                             color=colors['CO2'], linewidth=1, label='CO2吸收系数')[0]
            lines.append(line_co2)

        # 绘制CO吸收系数
        if 'CO' in individual_coefs:
            line_co = self.canvas.axes.plot(wavenumber, individual_coefs['CO'],
                                             color=colors['CO'], linewidth=1, label='CO吸收系数')[0]
            lines.append(line_co)

        # 绘制NO吸收系数
        if 'NO' in individual_coefs:
            line_no = self.canvas.axes.plot(wavenumber, individual_coefs['NO'],
                                            color=colors['NO'], linewidth=1, label='NO吸收系数')[0]
            lines.append(line_no)

        # 绘制N2O吸收系数
        if 'N2O' in individual_coefs:
            line_n2o = self.canvas.axes.plot(wavenumber, individual_coefs['N2O'],
                                            color=colors['N2O'], linewidth=1, label='N2O吸收系数')[0]
            lines.append(line_n2o)

        # 绘制NO吸收系数
        if 'NO2' in individual_coefs:
            line_no2 = self.canvas.axes.plot(wavenumber, individual_coefs['NO2'],
                                            color=colors['NO2'], linewidth=1, label='NO2吸收系数')[0]
            lines.append(line_no2)


        # 设置左轴标签
        self.canvas.axes.set_xlabel('波数 (cm$^{-1}$)')
        self.canvas.axes.set_ylabel('吸收系数 (cm$^{-1}$)', color='black')
        self.canvas.axes.tick_params(axis='y', labelcolor='black')

        # 创建右轴绘制透射率（叠加显示）
        ax2 = self.canvas.axes.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('透射率', color=color)
        line_tr = ax2.plot(wavenumber, Tr, color=color, linestyle='-',
                           linewidth=1, alpha=0.8, label='总透射率')[0]
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1)  # 确保透射率范围正确
        lines.append(line_tr)

        # 添加图例
        labels = [line.get_label() for line in lines]
        self.canvas.axes.legend(lines, labels, loc='upper right')

        # 添加标题
        params = self.current_spectrum_results['params']
        self.canvas.axes.set_title(f'混合气体吸收光谱\n'
                                   f'T={params["T"]}K, p={params["p"]}atm, l={params["l"]}cm')

        # 添加网格
        self.canvas.axes.grid(True, alpha=0.3)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def update_spectrum_stats_text(self):
        """更新光谱结果统计文本"""
        if self.current_spectrum_results is None:
            return

        params = self.current_spectrum_results['params']
        OD = self.current_spectrum_results['OD']
        Tr = self.current_spectrum_results['Tr']
        Ab = self.current_spectrum_results['Ab']
        individual_ODs = self.current_spectrum_results['individual_ODs']

        # 计算各分子的吸收系数
        individual_coefs = {}
        for molecule_name, od in individual_ODs.items():
            coef_single = od / (self.current_spectrum_results['params']['p'] * 101325 /
                                (1.380649e-23 * self.current_spectrum_results['params']['T']) *
                                self.current_spectrum_results['params']['l'])
            individual_coefs[molecule_name] = coef_single

        # 找到透射率最小的位置
        min_Tr_idx = np.argmin(Tr)
        min_Tr_wavenumber = self.current_spectrum_results['wavenumber'][min_Tr_idx]

        text = f"""计算参数:
温度: {params['T']} K    压力: {params['p']} atm    光程: {params['l']} cm
波数范围: {params['start']} - {params['end']} cm⁻¹    分辨率: {params['resolution']} cm⁻¹

"""

        # 添加各分子的吸收系数统计
        for molecule_name, coef in individual_coefs.items():
            if len(coef) > 0:
                max_coef_mol = np.max(coef)
                min_coef_mol = np.min(coef)
                mean_coef_mol = np.mean(coef)
                max_coef_idx = np.argmax(coef)
                max_coef_wavenumber = self.current_spectrum_results['wavenumber'][max_coef_idx]

                text += f"{molecule_name}吸收系数:\n"
                text += f"最大值: {max_coef_mol:.2e} cm⁻¹ (位于 {max_coef_wavenumber:.4f} cm⁻¹)\n"
                text += f"最小值: {min_coef_mol:.2e} cm⁻¹\n"
                text += f"平均值: {mean_coef_mol:.2e} cm⁻¹\n\n"

        text += f"""透射率:
最小值: {np.min(Tr):.6f} (位于 {min_Tr_wavenumber:.4f} cm⁻¹)
最大值: {np.max(Tr):.6f}
平均值: {np.mean(Tr):.6f}

光学深度:
最大值: {np.max(OD):.6f}
最小值: {np.min(OD):.6f}

吸收率:
最大值: {np.max(Ab):.6f}
最小值: {np.min(Ab):.6f}

数据点数: {len(self.current_spectrum_results['wavenumber'])}
"""
        self.stats_text.setText(text)

def main():
    app = QApplication(sys.argv)

        # 设置应用程序信息
    app.setApplicationName("气体化学平衡与光谱联合模拟")
    app.setApplicationVersion("1.0")

    window = CombinedGasSpectrumGUI()
    window.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
