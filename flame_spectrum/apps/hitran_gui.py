# apps/hitran_gui.py

import sys
import os

# ================= 【关键修复 1】解决 ModuleNotFoundError =================
# 获取当前文件所在目录 (apps/)，并获取其上一级目录 (项目根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 

# 将项目根目录插入到 Python 模块搜索路径的最前面
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# =======================================================================

import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QGroupBox, QFileDialog,
                             QMessageBox, QTextEdit, QDoubleSpinBox,
                             QProgressBar, QSplitter, QTableWidget, QTableWidgetItem,
                             QAbstractItemView, QHeaderView, QFormLayout, QInputDialog)
from PyQt6.QtCore import Qt

# 导入分离后的组件
from gui.components import CalculationThread
from gui.spectrum_plotter import SpectrumCanvas

# 导入核心引擎
from core.hitran_spectrum import HitranSpectrum

class HitranGUI(QMainWindow):
    def __init__(self, initial_T=300.0, initial_P=1.0, db_path="", cantera_species=None, q_folder=None):
        super().__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.db_path = db_path if db_path else os.path.join(project_root, 'hitran_database')
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)
        
        self.cantera_species = cantera_species if cantera_species else {}
        self.global_min_wn = float('inf')
        self.global_max_wn = 0.0
        self.calc_thread = None

        # 1. 先确定 Q 文件夹路径，但不设置控件
        if q_folder:
            self.q_folder = q_folder
        else:
            self.q_folder = os.path.join(self.db_path, 'Q')
            if not os.path.exists(self.q_folder):
                self.q_folder = self.db_path

        # 2. 构建界面（此时所有控件被创建）
        self.init_ui(initial_T, initial_P)

        # 3. 界面就绪后再初始化引擎和设置控件文本
        self.hitran_engine = HitranSpectrum(q_folder=self.q_folder)
        self.q_folder_edit.setText(self.q_folder)  # 现在可以安全调用


        
    def init_ui(self, T, P):
        self.setWindowTitle('HITRAN 混合气体光谱仿真工具 刘训臣')
        self.setGeometry(100, 100, 1400, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        main_layout.addWidget(self.create_control_panel(T, P), 1)
        main_layout.addWidget(self.create_right_panel(), 2)

    def create_control_panel(self, T, P):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # ---- 多文件表格 (文件名 + 浓度) ----
        file_group = QGroupBox("HITRAN 文件列表 (浓度可调)")
        file_layout = QVBoxLayout()
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(2)
        self.file_table.setHorizontalHeaderLabels(["PAR 文件", "浓度 (ppm)"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.file_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | 
                                        QAbstractItemView.EditTrigger.EditKeyPressed)
        file_layout.addWidget(self.file_table)

        btn_layout = QHBoxLayout()
        self.btn_add_par = QPushButton("添加 .par 文件")
        self.btn_add_par.clicked.connect(self.add_par_files)
        btn_layout.addWidget(self.btn_add_par)
        self.btn_remove_par = QPushButton("移除选中行")
        self.btn_remove_par.clicked.connect(self.remove_selected_rows)
        btn_layout.addWidget(self.btn_remove_par)
        self.btn_set_conc = QPushButton("设置浓度 (选中行)")
        self.btn_set_conc.clicked.connect(self.set_concentration_for_selected)
        btn_layout.addWidget(self.btn_set_conc)
        file_layout.addLayout(btn_layout)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # ---- 配分函数文件夹 ----
        q_group = QGroupBox("配分函数 (Q) 文件夹")
        q_layout = QHBoxLayout()
        default_q_path = self.q_folder   # 之前是 os.path.join(self.db_path, 'Q') 等，现在直接使用 self.q_folder
        self.q_folder_edit = QLineEdit(default_q_path)
        btn_q = QPushButton("浏览")
        btn_q.clicked.connect(self.select_q_folder)
        q_layout.addWidget(self.q_folder_edit)
        q_layout.addWidget(btn_q)
        q_group.setLayout(q_layout)
        layout.addWidget(q_group)

        # ---- 全局参数 ----
        param_group = QGroupBox("全局计算参数")
        param_layout = QFormLayout()
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(100, 5000)
        self.temp_spin.setValue(T)
        param_layout.addRow("温度 (K):", self.temp_spin)
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(0.001, 100)
        self.pressure_spin.setValue(P)
        param_layout.addRow("压力 (atm):", self.pressure_spin)
        self.path_spin = QDoubleSpinBox()
        self.path_spin.setRange(0.1, 10000)
        self.path_spin.setValue(100.0)
        param_layout.addRow("光程 (cm):", self.path_spin)
        self.omega_wing_spin = QDoubleSpinBox()
        self.omega_wing_spin.setRange(0.1, 500.0)
        self.omega_wing_spin.setValue(25.0)
        self.omega_wing_spin.setDecimals(1)
        param_layout.addRow("Omega Wing ($cm^{-1}$):", self.omega_wing_spin)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # ---- 波数范围 ----
        wn_group = QGroupBox("波数范围 (自动识别)")
        wn_layout = QFormLayout()
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 100000)
        self.start_spin.setValue(1000)
        self.start_spin.setDecimals(4)
        wn_layout.addRow("起始 ($cm^{-1}$):", self.start_spin)
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0, 100000)
        self.end_spin.setValue(5000)
        self.end_spin.setDecimals(4)
        wn_layout.addRow("结束 ($cm^{-1}$):", self.end_spin)
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.0001, 1.0)
        self.resolution_spin.setValue(0.01)
        self.resolution_spin.setDecimals(4)
        wn_layout.addRow("分辨率 ($cm^{-1}$):", self.resolution_spin)
        wn_group.setLayout(wn_layout)
        layout.addWidget(wn_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        btn_calc_layout = QHBoxLayout()
        self.calc_btn = QPushButton("开始计算混合光谱")
        self.calc_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.calc_btn.clicked.connect(self.start_calculation)
        btn_calc_layout.addWidget(self.calc_btn)
        self.clear_btn = QPushButton("清除图形")
        self.clear_btn.clicked.connect(self.clear_plot)
        btn_calc_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_calc_layout)

        info_db = QGroupBox("运行日志")
        info_db_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(120)
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

        # 使用分离出的 SpectrumCanvas，它包含 Figure、Canvas 和 Toolbar
        self.canvas = SpectrumCanvas(self, width=10, height=6, dpi=100)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas.toolbar)
        plot_layout.addWidget(self.canvas)
        plot_widget.setLayout(plot_layout)

        # 统计信息框
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        stats_widget.setLayout(stats_layout)
        stats_title = QLabel("计算结果统计")
        stats_title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 5px;")
        stats_layout.addWidget(stats_title)
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(plot_widget)
        splitter.addWidget(stats_widget)
        splitter.setSizes([600, 150])
        layout.addWidget(splitter)
        return panel

    # ---------- 文件与浓度管理 ----------
    def add_par_files(self):
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "选择 HITRAN .par 文件 (可多选)", self.db_path, "PAR Files (*.par);;All Files (*)"
        )
        if filenames:
            existing = [self.file_table.item(i, 0).text() for i in range(self.file_table.rowCount())]
            row = self.file_table.rowCount()
            for f in filenames:
                if f not in existing:
                    self.file_table.insertRow(row)
                    self.file_table.setItem(row, 0, QTableWidgetItem(f))
                    # 默认浓度：尝试从 cantera_species 获取，否则 10 ppm
                    base = os.path.splitext(os.path.basename(f))[0].split('_')[0].upper()
                    conc = 10.0  # ppm
                    if self.cantera_species:
                        for sp, frac in self.cantera_species.items():
                            if base in sp.upper() or sp.upper() in base:
                                conc = float(frac) * 1e6
                                break
                    self.file_table.setItem(row, 1, QTableWidgetItem(f"{conc:.2f}"))
                    self.update_wn_range_from_par(f)
                    row += 1
            if self.global_min_wn != float('inf'):
                self.start_spin.setValue(self.global_min_wn)
                self.end_spin.setValue(self.global_max_wn)
                self.log(f"自动更新波数范围: {self.global_min_wn:.2f} - {self.global_max_wn:.2f} cm^-1")

    def remove_selected_rows(self):
        for item in self.file_table.selectedItems():
            row = item.row()
            self.file_table.removeRow(row)

    def set_concentration_for_selected(self):
        """为选中的行设置浓度（ppm）"""
        selected = self.file_table.selectedItems()
        if not selected:
            QMessageBox.information(self, "提示", "请先选中要设置浓度的行")
            return
        rows = set(item.row() for item in selected)
        conc_str, ok = QInputDialog.getText(self, "设置浓度", "输入浓度 (ppm):", text="100.0")
        if ok:
            try:
                conc = float(conc_str)
                for r in rows:
                    self.file_table.setItem(r, 1, QTableWidgetItem(f"{conc:.2f}"))
            except ValueError:
                QMessageBox.warning(self, "错误", "请输入有效数字")

    def update_wn_range_from_par(self, par_file):
        try:
            with open(par_file, 'r') as f:
                for line in f:
                    if len(line) >= 15:
                        try:
                            wn = float(line[3:15].strip())
                            if wn < self.global_min_wn:
                                self.global_min_wn = wn
                            if wn > self.global_max_wn:
                                self.global_max_wn = wn
                        except ValueError:
                            continue
        except Exception as e:
            self.log(f"警告: 无法读取 {os.path.basename(par_file)} 的波数范围 ({e})")

    def select_q_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择配分函数(Q)文件夹", self.db_path)
        if folder:
            self.q_folder_edit.setText(folder)

    def log(self, msg):
        self.info_text.append(msg)

    # ---------- 计算 ----------
    def start_calculation(self):
        if self.file_table.rowCount() == 0:
            QMessageBox.warning(self, "警告", "请至少添加一个 .par 文件！")
            return

        q_folder = self.q_folder_edit.text()
        hitran_engine = HitranSpectrum(q_folder=q_folder)
        if os.path.exists(q_folder):
            self.log(f"配分函数文件夹: {q_folder}")
        else:
            self.log("警告: Q 文件夹不存在，使用简化公式")

        # 遍历表格行，添加分子
        for row in range(self.file_table.rowCount()):
            par_file = self.file_table.item(row, 0).text()
            conc_item = self.file_table.item(row, 1)
            conc_ppm = float(conc_item.text()) if conc_item else 10.0
            mole_fraction = conc_ppm * 1e-6

            base = os.path.splitext(os.path.basename(par_file))[0].split('_')[0].upper()
            try:
                hitran_engine.add_molecule(par_file, mole_fraction, name=base)
                self.log(f"加载: {base} (浓度: {conc_ppm:.2f} ppm)")
            except Exception as e:
                self.log(f"错误: 加载 {os.path.basename(par_file)} 失败 -> {str(e)}")
                return

        if not hitran_engine.molecules:
            QMessageBox.critical(self, "错误", "没有加载任何分子！")
            return

        # 收集计算参数
        T = self.temp_spin.value()
        p = self.pressure_spin.value()
        L = self.path_spin.value()
        start = self.start_spin.value()
        end = self.end_spin.value()
        resolution = self.resolution_spin.value()
        omega_wing = self.omega_wing_spin.value()

        self.calc_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log("开始计算...")

        # 使用分离出的 CalculationThread
        self.calc_thread = CalculationThread(
            hitran_engine, T, p, L, start, end, resolution, omega_wing
        )
        self.calc_thread.finished.connect(self.on_calculation_finished)
        self.calc_thread.error.connect(self.on_calculation_error)
        self.calc_thread.start()

    def on_calculation_finished(self, results):
        self.progress_bar.setVisible(False)
        self.calc_btn.setEnabled(True)
        self.log("计算完成，绘图...")

        # 直接调用画布的绘图方法
        self.canvas.plot_mixture(
            wavenumber=results['wavenumber'],
            total_coef=results['total_coef'],
            individual_coefs=results['individual_coefs'],
            transmittance=results['Tr'],
            params=results['params'],
            title="Mixture Spectrum",
            grid_density='fine',
            show_wavelength=True
        )

        # 更新统计信息
        params = results['params']
        T = params.get('T', '?')
        p = params.get('p', '?')
        L = params.get('l', '?')
        stats = f"<b>参数:</b> T={T}K, P={p}atm, L={L}cm<br><b>分子:</b><br>"
        for mol_name in results['individual_coefs'].keys():
            stats += f"- {mol_name}<br>"
        self.stats_text.setHtml(stats)

    def on_calculation_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.calc_btn.setEnabled(True)
        QMessageBox.critical(self, "计算错误", error_msg)
        self.log(f"错误: {error_msg}")

    def clear_plot(self):
        self.canvas.clear_all()
        self.canvas.draw()
        self.stats_text.clear()

    def auto_load_molecules(self, molecule_dict):
        """
        自动批量添加分子并设置浓度，无需用户手动浏览文件。
        molecule_dict: {分子名: (par_file_path, mole_fraction)}
        """
        for mol_name, (par_file, conc) in molecule_dict.items():
            if not os.path.exists(par_file):
                self.log(f"跳过 {mol_name}：文件不存在 ({par_file})")
                continue

            # 添加分子到引擎（如果引擎尚未初始化，此处会出错，因此 __init__ 中已提前创建）
            try:
                self.hitran_engine.add_molecule(par_file, conc, name=mol_name)
            except Exception as e:
                self.log(f"加载 {mol_name} 失败: {e}")
                continue

            # 更新文件表格（与 add_par_files 中的逻辑一致）
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            self.file_table.setItem(row, 0, QTableWidgetItem(par_file))
            ppm = conc * 1e6
            self.file_table.setItem(row, 1, QTableWidgetItem(f"{ppm:.2f}"))
            self.update_wn_range_from_par(par_file)
            self.log(f"自动加载: {mol_name} ({os.path.basename(par_file)}) 浓度: {ppm:.2f} ppm")

        # 最后统一更新波数范围控件
        if self.global_min_wn != float('inf'):
            self.start_spin.setValue(self.global_min_wn)
            self.end_spin.setValue(self.global_max_wn)        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    test_species = {'H2O': 0.15, 'CO2': 0.08, 'CO': 0.005}
    window = HitranGUI(initial_T=1500.0, initial_P=1.0, cantera_species=test_species)
    window.show()
    sys.exit(app.exec())
