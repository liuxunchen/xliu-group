# flame_spectrum.py
import sys
import os
import numpy as np
import cantera as ct
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QTextEdit, QDoubleSpinBox,
                             QGroupBox, QMessageBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QSplitter, QComboBox, QTabWidget, QCheckBox,
                             QProgressBar, QFileDialog, QFrame, QScrollArea, QDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# 导入HitranSpectrum类（多分子版本）
sys.path.insert(0, 'voigt_simulation')
from hitran_spectrum_dual import HitranSpectrum

# 强制使用系统自带中文字体（Windows 微软雅黑）
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False
rcParams['text.usetex'] = False
rcParams['mathtext.fontset'] = 'stix'


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
    info = pyqtSignal(str)  # 用于发送信息消息

    def __init__(self, hitran, params, concentrations):
        super().__init__()
        self.hitran = hitran
        self.params = params
        self.concentrations = concentrations  # 新增：存储浓度信息

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

                # 计算各分子的单独透射率
                individual_Trs = {}
                for molecule_name, od in individual_ODs.items():
                    individual_Trs[molecule_name] = np.exp(-od)

                # 计算对应的波长（单位：微米）
                wavelength_micron = 10000.0 / wavenumber  # 波数(cm⁻¹)转波长(微米)

            results = {
                'wavenumber': wavenumber,
                'wavelength_micron': wavelength_micron,  # 波长数据
                'total_coef': total_coef,
                'OD': OD,
                'Ab': Ab,
                'Tr': Tr,
                'individual_ODs': individual_ODs,
                'individual_Trs': individual_Trs,  # 存储各分子透射率
                'params': self.params,
                'concentrations': self.concentrations  # 新增：保存浓度信息
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

    def detect_par_file_format(self, file_path):
        """检测PAR文件格式"""
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()

            if not first_line:
                return "unknown"

            # 分析格式特征
            parts = first_line.split()

            if len(parts) >= 2:
                # 检查波数特征
                wavenumber_str = parts[1]

                # 判断是否是科学计数法或非常小的数（压缩格式）
                if 'E' in wavenumber_str.upper() or float(wavenumber_str) < 10:
                    return "compressed"  # 压缩格式（如H2O文件）
                else:
                    return "standard"  # 标准格式（如CO2文件）

            return "unknown"

        except Exception as e:
            print(f"检测文件格式失败: {e}")
            return "unknown"

    def parse_par_file_range(self, par_file_path):
        """解析par文件的波数范围 - 自动检测格式"""
        try:
            # 首先检测文件格式
            file_format = self.detect_par_file_format(par_file_path)
            print(f"检测到文件格式: {file_format}")

            min_wavenumber = float('inf')
            max_wavenumber = float('-inf')
            line_count = 0

            with open(par_file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    # 跳过注释行
                    if line.startswith('#') or line.startswith('!'):
                        continue

                    wavenumber = None

                    if file_format == "compressed":
                        # H2O压缩格式：波数在第2-8列
                        if len(line) >= 8:
                            try:
                                wavenumber_str = line[1:8].strip()
                                if wavenumber_str:
                                    wavenumber = float(wavenumber_str)
                            except ValueError:
                                pass

                    elif file_format == "standard":
                        # CO2标准格式：波数在第2个字段
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                wavenumber_str = parts[1]
                                wavenumber = float(wavenumber_str)
                            except ValueError:
                                pass

                    else:
                        # 未知格式，尝试多种方法
                        parts = line.split()
                        if len(parts) >= 2:
                            for part in parts[1:3]:  # 尝试第2和第3个字段
                                try:
                                    wavenumber = float(part)
                                    if 0 <= wavenumber <= 10000:
                                        break
                                except ValueError:
                                    continue

                    # 验证并记录波数
                    if wavenumber is not None and 0 <= wavenumber <= 10000:
                        min_wavenumber = min(min_wavenumber, wavenumber)
                        max_wavenumber = max(max_wavenumber, wavenumber)
                        line_count += 1

                        # 调试：显示前几行
                        if line_count <= 3:
                            print(f"行{line_num}: 格式={file_format}, 波数={wavenumber}")

            print(f"解析完成: {line_count}行，范围: {min_wavenumber} - {max_wavenumber} cm⁻¹")

            if line_count > 0 and min_wavenumber != float('inf') and max_wavenumber != float('-inf'):
                return min_wavenumber, max_wavenumber
            else:
                return None

        except Exception as e:
            print(f"解析par文件范围失败: {e}")
            import traceback
            traceback.print_exc()
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
        self.path_spin.setRange(0.01, 10000)
        self.path_spin.setValue(10.0)
        self.path_spin.setSingleStep(10)
        path_layout.addWidget(self.path_spin)
        layout.addLayout(path_layout)

        # 波数范围组（网格细化）
        wavenumber_group = QGroupBox("波数范围与网格设置")
        wavenumber_layout = QVBoxLayout()

        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("起始:"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(400, 5000)
        self.start_spin.setValue(2000)
        self.start_spin.setDecimals(1)
        self.start_spin.setSingleStep(10)
        self.start_spin.setSuffix(" cm⁻¹")
        range_layout.addWidget(self.start_spin)

        range_layout.addWidget(QLabel("结束:"))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(400, 5000)
        self.end_spin.setValue(2300)
        self.end_spin.setDecimals(1)
        self.end_spin.setSingleStep(10)
        self.end_spin.setSuffix(" cm⁻¹")
        range_layout.addWidget(self.end_spin)

        range_layout.addWidget(QLabel("分辨率:"))
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.001, 1.0)
        self.resolution_spin.setValue(0.1)
        self.resolution_spin.setDecimals(3)
        self.resolution_spin.setSingleStep(0.01)
        self.resolution_spin.setSuffix(" cm⁻¹")
        range_layout.addWidget(self.resolution_spin)

        wavenumber_layout.addLayout(range_layout)

        # omega_wing参数
        wing_layout = QHBoxLayout()
        wing_layout.addWidget(QLabel("谱线计算域倍数:"))
        self.omega_wing_spin = QDoubleSpinBox()
        self.omega_wing_spin.setRange(5, 200)
        self.omega_wing_spin.setValue(30)
        self.omega_wing_spin.setSingleStep(5)
        wing_layout.addWidget(self.omega_wing_spin)
        wavenumber_layout.addLayout(wing_layout)

        # 波长显示设置
        wavelength_layout = QHBoxLayout()
        wavelength_layout.addWidget(QLabel("波长显示:"))
        self.wavelength_check = QCheckBox("在图形顶部显示波长")
        self.wavelength_check.setChecked(True)
        wavelength_layout.addWidget(self.wavelength_check)
        wavenumber_layout.addLayout(wavelength_layout)

        # 添加重置按钮
        reset_layout = QHBoxLayout()
        self.reset_wavenumber_btn = QPushButton("重置为默认值")
        self.reset_wavenumber_btn.clicked.connect(self.reset_wavenumber_defaults)
        reset_layout.addWidget(self.reset_wavenumber_btn)
        reset_layout.addStretch()
        wavenumber_layout.addLayout(reset_layout)

        # 网格细化说明
        warning_label = QLabel(
            "注意：更小的分辨率和更大的计算域倍数会显著增加计算时间")
        warning_label.setStyleSheet(
            "color: #d35400; font-size: 10px; padding: 5px; background-color: #fff3cd; border-radius: 3px;")
        wavenumber_layout.addWidget(warning_label)

        wavenumber_group.setLayout(wavenumber_layout)
        layout.addWidget(wavenumber_group)

        # 透射率显示选择
        transmittance_group = QGroupBox("透射率显示设置")
        transmittance_layout = QVBoxLayout()

        # 透射率类型选择
        trans_type_layout = QHBoxLayout()
        trans_type_layout.addWidget(QLabel("透射率类型:"))
        self.transmittance_combo = QComboBox()
        self.transmittance_combo.addItems(
            ["总透射率", "H2O透射率", "CO2透射率", "CO透射率", "NO透射率", "N2O透射率", "NO2透射率"])
        trans_type_layout.addWidget(self.transmittance_combo)
        transmittance_layout.addLayout(trans_type_layout)

        # 显示选项
        self.show_abs_check = QCheckBox("显示吸收系数")
        self.show_abs_check.setChecked(True)
        transmittance_layout.addWidget(self.show_abs_check)

        self.show_trans_check = QCheckBox("显示透射率")
        self.show_trans_check.setChecked(True)
        transmittance_layout.addWidget(self.show_trans_check)

        transmittance_group.setLayout(transmittance_layout)
        layout.addWidget(transmittance_group)

        # 图形显示设置
        plot_settings_group = QGroupBox("图形显示设置")
        plot_settings_layout = QVBoxLayout()

        # 图形网格设置
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("图形网格密度:"))
        self.grid_density_combo = QComboBox()
        self.grid_density_combo.addItems(["精细", "中等", "稀疏"])
        self.grid_density_combo.setCurrentText("精细")
        self.grid_density_combo.currentTextChanged.connect(self.on_grid_density_changed)
        grid_layout.addWidget(self.grid_density_combo)
        plot_settings_layout.addLayout(grid_layout)

        # 图形尺寸设置
        figsize_layout = QHBoxLayout()
        figsize_layout.addWidget(QLabel("图形尺寸:"))
        self.figsize_combo = QComboBox()
        self.figsize_combo.addItems(["大(12x8)", "中(10x6)", "小(8x5)"])
        self.figsize_combo.setCurrentText("中(10x6)")
        self.figsize_combo.currentTextChanged.connect(self.on_figsize_changed)
        figsize_layout.addWidget(self.figsize_combo)
        plot_settings_layout.addLayout(figsize_layout)

        plot_settings_group.setLayout(plot_settings_layout)
        layout.addWidget(plot_settings_group)

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

    def reset_wavenumber_defaults(self):
        """重置波数设置为默认值"""
        self.start_spin.setValue(4650)
        self.end_spin.setValue(5410)
        self.resolution_spin.setValue(0.01)
        self.omega_wing_spin.setValue(30)

        QMessageBox.information(self, "重置", "波数设置已重置为默认值")

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
        self.stats_text.setMaximumHeight(300)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)

        # 设置分割器
        splitter.addWidget(plot_widget)
        splitter.addWidget(stats_widget)
        splitter.setSizes([600, 400])

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

        # 存储手动浓度控件的字典（单位：ppm）
        self.manual_conc_spins = {}

        # 初始为选中的分子创建浓度输入框（单位：ppm）
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
        """为手动组添加浓度输入框（单位：ppm）"""
        if molecule_name not in self.manual_conc_spins:
            conc_layout = QHBoxLayout()
            conc_layout.addWidget(QLabel(f"{molecule_name}浓度 (ppm):"))
            spin = QDoubleSpinBox()
            spin.setRange(0, 1000000)
            spin.setValue(100000)
            spin.setDecimals(2)
            spin.setSuffix(" ppm")
            spin.setSingleStep(1000)
            spin.setToolTip(f"{molecule_name}浓度，单位：ppm（百万分之一）")
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

            # 为手动组添加浓度输入框（单位：ppm）
            self.add_manual_concentration_spin(molecule_name, self.manual_layout)

            # 启用删除按钮
            self.remove_molecule_btn.setEnabled(len(self.selected_molecules) > 1)

            # 更新透射率选择框
            self.update_transmittance_combo()

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

            # 更新透射率选择框
            self.update_transmittance_combo()

            if dialog:
                dialog.accept()

            # 更新信息显示
            self.spectrum_info_text.append(f"已删除分子: {molecule_name}")

    def update_transmittance_combo(self):
        """更新透射率选择框"""
        current_text = self.transmittance_combo.currentText()
        self.transmittance_combo.clear()

        # 添加总透射率选项
        self.transmittance_combo.addItem("总透射率")

        # 添加当前选中的分子的透射率选项
        for molecule in self.selected_molecules:
            self.transmittance_combo.addItem(f"{molecule}透射率")

        # 恢复之前的选项（如果还存在）
        for i in range(self.transmittance_combo.count()):
            if self.transmittance_combo.itemText(i) == current_text:
                self.transmittance_combo.setCurrentIndex(i)
                break

    def on_grid_density_changed(self, density):
        """图形网格密度改变"""
        if self.current_spectrum_results is not None:
            self.update_spectrum_plot()

    def on_figsize_changed(self, size_text):
        """图形尺寸改变"""
        sizes = {
            "大(12x8)": (12, 8),
            "中(10x6)": (10, 6),
            "小(8x5)": (8, 5)
        }

        if size_text in sizes:
            width, height = sizes[size_text]
            self.canvas.fig.set_size_inches(width, height)
            if self.current_spectrum_results is not None:
                self.update_spectrum_plot()
            self.canvas.draw()

    def on_molecule_file_changed(self, molecule_name, text):
        """分子文件选择变化"""
        if text and text != "未找到文件":
            self.spectrum_info_text.append(f"{molecule_name}文件已选择: {text}")

    def auto_set_wavenumber_range(self):
        """自动设置波数范围 - 处理混合格式文件"""
        ranges = []  # 存储各分子的波数范围
        molecule_info = []  # 存储分子信息

        # 检查所有选中的分子文件
        for molecule_name in self.selected_molecules:
            if molecule_name in self.molecule_widgets:
                widget = self.molecule_widgets[molecule_name]
                file_path = widget.file_combo.currentData()

                if file_path and os.path.exists(file_path):
                    self.spectrum_info_text.append(f"解析 {molecule_name} 文件...")

                    # 检测文件格式
                    file_format = self.detect_par_file_format(file_path)

                    # 解析波数范围
                    file_range = self.parse_par_file_range(file_path)

                    if file_range:
                        start, end = file_range
                        ranges.append((start, end))
                        molecule_info.append({
                            'name': molecule_name,
                            'format': file_format,
                            'start': start,
                            'end': end
                        })

                        self.spectrum_info_text.append(
                            f"{molecule_name} ({file_format}格式): {start:.6f} - {end:.6f} cm⁻¹"
                        )
                    else:
                        self.spectrum_info_text.append(f"警告：无法解析 {molecule_name} 文件范围")

        # 如果没有解析到任何范围，设置默认值
        if not ranges:
            self.set_default_wavenumber_range()
            return False

        # 分析所有分子的波数范围
        self.spectrum_info_text.append("\n=== 波数范围分析 ===")

        # 显示各分子的详细范围
        for info in molecule_info:
            self.spectrum_info_text.append(
                f"{info['name']}: {info['start']:.6f} - {info['end']:.6f} cm⁻¹ (格式: {info['format']})"
            )

        # 计算整体范围
        all_starts = [r[0] for r in ranges]
        all_ends = [r[1] for r in ranges]

        overall_min = min(all_starts)
        overall_max = max(all_ends)

        self.spectrum_info_text.append(f"\n总体范围: {overall_min:.6f} - {overall_max:.6f} cm⁻¹")

        # 检查是否有远红外数据（< 10 cm⁻¹）
        has_far_ir = any(r[0] < 10 for r in ranges)

        if has_far_ir:
            self.spectrum_info_text.append("检测到远红外数据（< 10 cm⁻¹）")

            # 如果有远红外数据，可能需要分情况处理
            # 方案1：如果同时有中红外数据，优先考虑中红外
            mid_ir_ranges = [r for r in ranges if r[0] > 100]

            if mid_ir_ranges:
                self.spectrum_info_text.append("同时检测到中红外数据，优先使用中红外范围")
                mid_ir_starts = [r[0] for r in mid_ir_ranges]
                mid_ir_ends = [r[1] for r in mid_ir_ranges]
                start_range = min(mid_ir_starts)
                end_range = max(mid_ir_ends)
            else:
                # 只有远红外数据
                self.spectrum_info_text.append("只有远红外数据，使用远红外范围")
                start_range = max(1, overall_min)  # 至少从1 cm⁻¹开始
                end_range = min(1000, overall_max)  # 限制到1000 cm⁻¹
        else:
            # 只有中红外数据
            start_range = overall_min
            end_range = overall_max

        # 增加边界
        range_width = end_range - start_range
        if range_width > 0:
            extension = range_width * 0.1  # 10%边界
            start_range = max(1, start_range - extension)
            end_range = end_range + extension
        else:
            # 如果范围太小，设置合理的最小宽度
            end_range = start_range + 100

        # 限制最大范围
        if end_range - start_range > 5000:
            end_range = start_range + 5000
            self.spectrum_info_text.append("范围太大，限制到5000 cm⁻¹宽度")

        # 设置UI控件
        self.start_spin.setValue(round(start_range, 2))
        self.end_spin.setValue(round(end_range, 2))

        self.spectrum_info_text.append(f"\n最终设置范围: {start_range:.2f} - {end_range:.2f} cm⁻¹")

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
                        # 显示分数和ppm两种单位
                        ppm_value = fraction * 1000000
                        combo.addItem(f"{species}: {fraction:.6f} ({ppm_value:.2f} ppm)", fraction)
                    combo.setCurrentIndex(0)
                else:
                    combo.addItem(f"未找到{molecule_name}", 0.0)

                # 同时更新手动组的ppm输入框（如果有该分子）
                if molecule_name in self.manual_conc_spins:
                    ppm_value = 0.0
                    if options:
                        # 取第一个匹配的浓度
                        ppm_value = options[0][1] * 1000000
                    self.manual_conc_spins[molecule_name].setValue(ppm_value)

        # 切换到光谱模拟标签页
        self.tab_widget.setCurrentIndex(1)
        QMessageBox.information(self, "成功", "气体平衡结果已传输到光谱模拟")

    # ==================== 新增：辅助方法 ====================

    def get_molecule_concentration(self, molecule_name):
        """获取指定分子的浓度（体积分数）"""
        if self.import_gas_check.isChecked():
            # 从导入的浓度获取
            if molecule_name in self.import_conc_combos:
                conc = self.import_conc_combos[molecule_name].currentData()
                return conc if conc is not None else 0.0
        else:
            # 从手动输入获取（ppm转分数）
            if molecule_name in self.manual_conc_spins:
                ppm_value = self.manual_conc_spins[molecule_name].value()
                return ppm_value / 1000000.0

        return 0.0

    def get_all_concentrations(self):
        """获取所有选中分子的浓度（字典形式）"""
        concentrations = {}
        for molecule_name in self.selected_molecules:
            concentrations[molecule_name] = self.get_molecule_concentration(molecule_name)
        return concentrations

    def calculate_absorption_coefficient(self, od, molecule_name):
        """计算吸收系数（单位：cm⁻¹）"""
        if self.current_spectrum_results is None:
            return np.zeros_like(od)

        params = self.current_spectrum_results['params']
        l = params['l']  # 光程 (cm)

        # 关键：OD 已经包含了浓度影响，所以吸收系数应该是：
        # α = OD / l
        absorption_coef = od / l

        return absorption_coef

    # 光谱模拟相关方法
    def on_import_gas_toggled(self, checked):
        """导入气体平衡结果复选框状态改变"""
        self.manual_temp_spin.setEnabled(not checked)
        self.manual_pressure_spin.setEnabled(not checked)
        for spin in self.manual_conc_spins.values():
            spin.setEnabled(not checked)

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
                    # 从导入的浓度下拉框获取浓度（保持为分数）
                    if molecule_name in self.import_conc_combos:
                        conc = self.import_conc_combos[molecule_name].currentData()
                    else:
                        conc = 0.0
                else:
                    # 从手动输入框获取浓度，将ppm转换为分数
                    if molecule_name in self.manual_conc_spins:
                        ppm_value = self.manual_conc_spins[molecule_name].value()
                        conc = ppm_value / 1000000.0  # 转换为分数
                    else:
                        conc = 0.1

                # 添加分子到光谱模拟器
                self.spectrum_simulator.add_molecule(file_path, concentration=conc,
                                                     molecule_name=molecule_name)

            # 显示分子信息
            self.spectrum_info_text.setText(self.spectrum_simulator.get_molecule_info())

            # 移除自动设置波数范围的调用
            # self.spectrum_info_text.append("正在自动设置波数范围...")
            # success = self.auto_set_wavenumber_range()
            # if success:
            #     self.spectrum_info_text.append("波数范围自动设置成功！")
            # else:
            #     self.spectrum_info_text.append("警告：无法自动设置波数范围，请手动设置")

            # 改为显示当前波数设置
            current_start = self.start_spin.value()
            current_end = self.end_spin.value()
            self.spectrum_info_text.append(f"\n当前波数设置: {current_start} - {current_end} cm⁻¹")
            self.spectrum_info_text.append("提示：请手动设置波数范围后开始计算")

            self.calculate_spectrum_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "分子数据加载成功！\n请手动设置波数范围后开始计算。")

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

            # 获取当前所有分子的浓度
            concentrations = self.get_all_concentrations()

            # 在单独的线程中执行计算，传入浓度信息
            self.calculation_thread = SpectrumCalculationThread(
                self.spectrum_simulator, params, concentrations
            )
            self.calculation_thread.finished.connect(self.on_spectrum_calculation_finished)
            self.calculation_thread.error.connect(self.on_spectrum_calculation_error)
            self.calculation_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {e}")
            self.calculate_spectrum_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def on_spectrum_calculation_finished(self, results):
        """光谱计算完成"""
        self.current_spectrum_results = results

        # 确保透射率计算正确：Tr = exp(-OD)
        results['Tr'] = np.exp(-results['OD'])

        # 计算各分子的单独透射率
        for molecule_name, od in results['individual_ODs'].items():
            results['individual_Trs'][molecule_name] = np.exp(-od)
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
        """更新光谱图形 - 改为柱状图显示"""
        if self.current_spectrum_results is None:
            return

        # 清除图形
        self.canvas.axes.clear()

        # 获取结果数据
        params = self.current_spectrum_results['params']
        wavenumber = self.current_spectrum_results['wavenumber']
        wavelength_micron = self.current_spectrum_results['wavelength_micron']
        individual_ODs = self.current_spectrum_results['individual_ODs']
        individual_Trs = self.current_spectrum_results['individual_Trs']
        Tr_total = self.current_spectrum_results['Tr']

        # 获取选择的透射率类型
        selected_transmittance = self.transmittance_combo.currentText()

        # 确定要显示的透射率数据
        if selected_transmittance == "总透射率":
            transmittance_data = Tr_total
            transmittance_label = "总透射率"
        else:
            transmittance_data = Tr_total
            transmittance_label = "总透射率"
            for molecule in self.selected_molecules:
                if selected_transmittance.startswith(molecule):
                    if molecule in individual_Trs:
                        transmittance_data = individual_Trs[molecule]
                        transmittance_label = f"{molecule}透射率"
                    break

        lines = []

        # 计算各分子的吸收系数（单位：cm⁻¹）
        individual_coefs = {}
        for molecule_name, od in individual_ODs.items():
            # 吸收系数 = OD / 光程
            absorption_coef = od / params['l']
            individual_coefs[molecule_name] = absorption_coef

        # 设置图形网格密度
        grid_density = self.grid_density_combo.currentText()
        if grid_density == "精细":
            grid_alpha = 0.2
            grid_linestyle = ':'
            grid_linewidth = 0.5
        elif grid_density == "中等":
            grid_alpha = 0.3
            grid_linestyle = '--'
            grid_linewidth = 0.7
        else:
            grid_alpha = 0.4
            grid_linestyle = '-'
            grid_linewidth = 0.8

        # 显示吸收系数（柱状图）
        if self.show_abs_check.isChecked():
            colors = {'H2O': 'tab:blue', 'CO2': 'tab:green', 'CO': 'tab:red',
                      'NO': 'tab:purple', 'N2O': 'tab:pink', 'NO2': 'tab:brown'}

            # 计算柱状图宽度
            if len(wavenumber) > 1:
                # 根据数据点间隔确定柱宽
                wavenumber_diff = np.diff(wavenumber)
                avg_diff = np.mean(wavenumber_diff)
                bar_width = avg_diff * 0.8  # 柱宽为平均间隔的80%
            else:
                bar_width = 0.1  # 默认值

            for molecule_name, coef in individual_coefs.items():
                if molecule_name in colors and np.max(np.abs(coef)) > 1e-10:
                    # 使用柱状图代替曲线图
                    bars = self.canvas.axes.bar(
                        wavenumber, coef,
                        width=bar_width,
                        color=colors[molecule_name],
                        alpha=0.7,
                        edgecolor=colors[molecule_name],
                        linewidth=0.5,
                        label=f'{molecule_name}吸收系数'
                    )
                    # 存储第一个柱状作为图例项
                    if bars:
                        lines.append(bars[0])

        # 设置x轴标签（底部：波数，顶部：波长）
        if self.wavelength_check.isChecked():
            ax_top = self.canvas.axes.twiny()
            self.canvas.axes.set_xlabel('波数 (cm$^{-1}$)', fontsize=11, labelpad=12)
            ax_top.set_xlabel('波长 ($\mu$m)', fontsize=11, labelpad=12)
            ax_top.set_xlim(self.canvas.axes.get_xlim())  # 修正这里！

            # 设置顶部x轴的刻度
            wavenumber_ticks = self.canvas.axes.get_xticks()
            wavenumber_ticks = wavenumber_ticks[(wavenumber_ticks >= wavenumber.min()) &
                                                (wavenumber_ticks <= wavenumber.max())]
            wavelength_ticks = 10000.0 / wavenumber_ticks
            ax_top.set_xticks(wavenumber_ticks)
            ax_top.set_xticklabels([f'{w:.4f}' for w in wavelength_ticks], fontsize=10)
            ax_top.tick_params(axis='x', which='major', size=6, width=1.5, direction='in', top=True)
            ax_top.tick_params(axis='x', which='minor', size=3, width=1, direction='in', top=True)
            ax_top.grid(True, alpha=grid_alpha * 0.7, linestyle=':', linewidth=0.3)
        else:
            self.canvas.axes.set_xlabel('波数 (cm$^{-1}$)', fontsize=11)

        # 设置y轴标签
        if self.show_abs_check.isChecked():
            self.canvas.axes.set_ylabel('吸收系数 (cm$^{-1}$)', color='black', fontsize=11)
            self.canvas.axes.tick_params(axis='y', labelcolor='black', labelsize=10)
        else:
            self.canvas.axes.set_ylabel('透射率', color='tab:orange', fontsize=11)
            self.canvas.axes.tick_params(axis='y', labelcolor='tab:orange', labelsize=10)

        # 显示透射率（柱状图）
        if self.show_trans_check.isChecked():
            if self.show_abs_check.isChecked():
                ax2 = self.canvas.axes.twinx()
                color = 'tab:orange'
                ax2.set_ylabel('透射率', color=color, fontsize=11)

                # 计算柱宽
                if len(wavenumber) > 1:
                    wavenumber_diff = np.diff(wavenumber)
                    avg_diff = np.mean(wavenumber_diff)
                    bar_width = avg_diff * 0.8
                else:
                    bar_width = 0.1

                # 使用柱状图显示透射率
                bars_tr = ax2.bar(
                    wavenumber, transmittance_data,
                    width=bar_width,
                    color=color,
                    alpha=0.5,
                    edgecolor=color,
                    linewidth=0.5,
                    label=transmittance_label
                )
                if bars_tr:
                    lines.append(bars_tr[0])

                ax2.tick_params(axis='y', labelcolor=color, labelsize=10)
                ax2.set_ylim(0, 1)
            else:
                color = 'tab:orange'

                # 计算柱宽
                if len(wavenumber) > 1:
                    wavenumber_diff = np.diff(wavenumber)
                    avg_diff = np.mean(wavenumber_diff)
                    bar_width = avg_diff * 0.8
                else:
                    bar_width = 0.1

                # 使用柱状图显示透射率
                bars_tr = self.canvas.axes.bar(
                    wavenumber, transmittance_data,
                    width=bar_width,
                    color=color,
                    alpha=0.5,
                    edgecolor=color,
                    linewidth=0.5,
                    label=transmittance_label
                )
                if bars_tr:
                    lines.append(bars_tr[0])

                self.canvas.axes.set_ylim(0, 1)

        # 添加图例
        if lines:
            labels = [line.get_label() for line in lines]
            legend = self.canvas.axes.legend(
                lines, labels,
                loc='upper right',
                fontsize=10,
                framealpha=0.9
            )
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(0.5)

        # 添加标题（包含浓度信息）
        conc_info = []
        for molecule_name in self.selected_molecules:
            conc = self.get_molecule_concentration(molecule_name)
            if conc > 1e-6:  # 只显示浓度较大的分子
                ppm_value = conc * 1000000
                conc_info.append(f"{molecule_name}: {ppm_value:.1f}ppm")

        conc_str = ", ".join(conc_info)

        title = f'混合气体吸收光谱（柱状图）\nT={params["T"]}K, p={params["p"]}atm, l={params["l"]}cm'
        if conc_str:
            title += f'\n浓度: {conc_str}'

        if self.wavelength_check.isChecked():
            wavelength_min = 10000.0 / wavenumber.max()
            wavelength_max = 10000.0 / wavenumber.min()
            title += f'\n波长范围: {wavelength_min:.4f} - {wavelength_max:.4f} μm'

        self.canvas.axes.set_title(title, fontsize=12, fontweight='bold', pad=20)

        # 添加网格
        self.canvas.axes.grid(True, alpha=grid_alpha, linestyle=grid_linestyle,
                              linewidth=grid_linewidth, zorder=0)

        # 设置x轴刻度
        x_min, x_max = wavenumber.min(), wavenumber.max()
        x_range = x_max - x_min
        if x_range > 0:
            if x_range < 100:
                major_ticks = np.arange(np.floor(x_min / 5) * 5, np.ceil(x_max / 5) * 5, 5)
                minor_ticks = np.arange(x_min, x_max, 1)
            elif x_range < 500:
                major_ticks = np.arange(np.floor(x_min / 20) * 20, np.ceil(x_max / 20) * 20, 20)
                minor_ticks = np.arange(x_min, x_max, 5)
            else:
                major_ticks = np.arange(np.floor(x_min / 50) * 50, np.ceil(x_max / 50) * 50, 50)
                minor_ticks = np.arange(x_min, x_max, 10)

            self.canvas.axes.set_xticks(major_ticks)
            self.canvas.axes.set_xticks(minor_ticks, minor=True)
            self.canvas.axes.xaxis.set_tick_params(which='major', size=6, width=1.5)
            self.canvas.axes.xaxis.set_tick_params(which='minor', size=3, width=1)

        # 设置y轴刻度
        self.canvas.axes.yaxis.set_tick_params(which='major', size=6, width=1.5)
        self.canvas.axes.yaxis.set_tick_params(which='minor', size=3, width=1)

        # 设置背景色
        self.canvas.axes.set_facecolor('#f8f9fa')
        self.canvas.fig.set_facecolor('#f1f3f4')

        # 调整布局
        self.canvas.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def update_spectrum_stats_text(self):
        """更新光谱结果统计文本（修复吸收系数计算）"""
        if self.current_spectrum_results is None:
            return

        params = self.current_spectrum_results['params']
        OD = self.current_spectrum_results['OD']
        Tr = self.current_spectrum_results['Tr']
        Ab = self.current_spectrum_results['Ab']
        individual_ODs = self.current_spectrum_results['individual_ODs']
        individual_Trs = self.current_spectrum_results['individual_Trs']

        # 获取波数范围信息
        wavenumber = self.current_spectrum_results['wavenumber']

        # 获取浓度信息
        conc_info = []
        for molecule_name in self.selected_molecules:
            conc = self.get_molecule_concentration(molecule_name)
            if conc > 1e-6:
                ppm_value = conc * 1000000
                conc_info.append(f"{molecule_name}: {ppm_value:.1f}ppm")

        conc_str = "，".join(conc_info)

        # 计算各分子的吸收系数
        individual_coefs = {}
        for molecule_name, od in individual_ODs.items():
            # 直接使用 od / l 作为吸收系数
            absorption_coef = od / params['l']
            individual_coefs[molecule_name] = absorption_coef
        # 构建统计文本
        text = f"""计算参数:
温度: {params['T']} K    压力: {params['p']} atm    光程: {params['l']} cm
波数范围: {params['start']:.2f} - {params['end']:.2f} cm⁻¹
对应波长范围: {10000.0 / params['end']:.4f} - {10000.0 / params['start']:.4f} μm
分辨率: {params['resolution']} cm⁻¹
谱线计算域倍数: {params['omega_wing']}
浓度: {conc_str if conc_info else '未设置浓度'}

"""

        # 添加各分子的吸收系数统计
        for molecule_name, coef in individual_coefs.items():
            if len(coef) > 0 and np.max(np.abs(coef)) > 1e-10:
                max_coef_mol = np.max(coef)
                min_coef_mol = np.min(coef[coef > 0]) if np.any(coef > 0) else 0
                mean_coef_mol = np.mean(coef[coef > 0]) if np.any(coef > 0) else 0

                if max_coef_mol > 0:
                    max_coef_idx = np.argmax(coef)
                    max_coef_wavenumber = wavenumber[max_coef_idx]
                    max_coef_wavelength = 10000.0 / max_coef_wavenumber

                    text += f"{molecule_name}吸收系数:\n"
                    text += f"  最大值: {max_coef_mol:.2e} cm⁻¹ (位于 {max_coef_wavenumber:.4f} cm⁻¹ / {max_coef_wavelength:.4f} μm)\n"
                    text += f"  最小值: {min_coef_mol:.2e} cm⁻¹\n"
                    text += f"  平均值: {mean_coef_mol:.2e} cm⁻¹\n"

                    # 添加透射率统计
                    if molecule_name in individual_Trs:
                        tr_mol = individual_Trs[molecule_name]
                        min_tr_mol = np.min(tr_mol)
                        max_tr_mol = np.max(tr_mol)
                        mean_tr_mol = np.mean(tr_mol)
                        min_tr_idx = np.argmin(tr_mol)
                        min_tr_wavenumber = wavenumber[min_tr_idx]
                        min_tr_wavelength = 10000.0 / min_tr_wavenumber

                        text += f"{molecule_name}透射率:\n"
                        text += f"  最小值: {min_tr_mol:.6f} (位于 {min_tr_wavenumber:.4f} cm⁻¹ / {min_tr_wavelength:.4f} μm)\n"
                        text += f"  最大值: {max_tr_mol:.6f}\n"
                        text += f"  平均值: {mean_tr_mol:.6f}\n\n"
                    else:
                        text += "\n"

        # 总透射率统计
        if len(Tr) > 0:
            min_tr_idx = np.argmin(Tr)
            min_tr_wavenumber = wavenumber[min_tr_idx]
            min_tr_wavelength = 10000.0 / min_tr_wavenumber

            text += f"""总透射率:
最小值: {np.min(Tr):.6f} (位于 {min_tr_wavenumber:.4f} cm⁻¹ / {min_tr_wavelength:.4f} μm)
最大值: {np.max(Tr):.6f}
平均值: {np.mean(Tr):.6f}

光学深度:
最大值: {np.max(OD):.6f}
最小值: {np.min(OD):.6f}

吸收率:
最大值: {np.max(Ab):.6f}
最小值: {np.min(Ab):.6f}

数据点数: {len(wavenumber)}
波数步长: {wavenumber[1] - wavenumber[0]:.6f} cm⁻¹
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
