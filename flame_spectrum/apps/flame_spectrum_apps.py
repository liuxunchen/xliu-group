import sys
import os
import cantera as ct

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGroupBox, QLabel, QLineEdit, 
                             QPushButton, QComboBox, QMessageBox, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QCheckBox,
                             QScrollArea, QSplitter)
from PyQt6.QtCore import Qt

# 动态添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# 将 apps/ 目录也加入路径，确保 `from hitran_gui` 等导入在任何调用方式下都能工作
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.flame_simulator import GasCompositionSimulator
from hitran_gui import HitranGUI 
from core.hitran_spectrum import HitranSpectrum
        
class FlameSpectrumApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cantera Flame Equilibrium + HITRAN Launcher (PyQt6)")
        self.resize(1100, 750)
        
        self.simulator = GasCompositionSimulator()
        self.simulator.initialize_gas('gri30.yaml')

        # 设置 hitran 数据库目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.hitran_db_dir = os.path.join(project_root, 'hitran_database')

        self.hitran_window = None 
        self.current_results = None
        self.init_ui()

        self.hitran_iso = HitranSpectrum.ISO
        # 构建分子名称 -> 分子ID 的映射（取第一个出现的ID，因为同一个名称可能有多条记录，但ID相同）
        self.name_to_mol_id = {}
        for (mol_id, iso_id), values in self.hitran_iso.items():
            name = values[4]  # 第5列是分子名称，如 'H2O', 'CO2'
            if name not in self.name_to_mol_id:
                self.name_to_mol_id[name] = mol_id

        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ================= 左侧控制面板 =================
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        scroll.setWidget(left_panel)

        flame_group = QGroupBox("1. Flame Equilibrium (Cantera)")
        flame_layout = QVBoxLayout()
        
        # 燃料
        flame_layout.addWidget(QLabel("Fuel:"))
        self.fuel_combo = QComboBox()
        self.fuel_combo.addItems(['H2', 'CH4', 'C2H6', 'C3H8', 'CO'])
        flame_layout.addWidget(self.fuel_combo)
        
        # 氧化剂下拉选项
        flame_layout.addWidget(QLabel("Oxidizer:"))
        self.oxidizer_combo = QComboBox()
        self.oxidizer_combo.addItems([
            "Air (O2:1.0 N2:3.76)", 
            "Pure O2 (O2:1.0)", 
            "Oxy-fuel (O2:1.0 CO2:2.0)",
            "Custom (自定义)"
        ])
        self.oxidizer_combo.currentTextChanged.connect(self.on_oxidizer_changed)
        flame_layout.addWidget(self.oxidizer_combo)

        self.oxidizer_edit = QLineEdit("O2:1.0 N2:3.76")
        self.oxidizer_edit.setPlaceholderText("e.g., O2:1.0 N2:3.76")
        flame_layout.addWidget(self.oxidizer_edit)

        # 初始状态
        state_layout = QHBoxLayout()
        state_layout.addWidget(QLabel("T_init (K):"))
        self.t_init_edit = QLineEdit("300.0")
        state_layout.addWidget(self.t_init_edit)
        state_layout.addWidget(QLabel("P_init (atm):"))
        self.p_init_edit = QLineEdit("1.0")
        state_layout.addWidget(self.p_init_edit)
        flame_layout.addLayout(state_layout)
        
        # 当量比
        flame_layout.addWidget(QLabel("Equivalence Ratio (Φ):"))
        self.phi_edit = QLineEdit("1.0")
        flame_layout.addWidget(self.phi_edit)

        # 平衡方法
        flame_layout.addWidget(QLabel("Equilibrate Method:"))
        self.eq_method_combo = QComboBox()
        self.eq_method_combo.addItems(["HP", "TP", "SP", "SV", "TV", "UV"])
        flame_layout.addWidget(self.eq_method_combo)
        
        self.calc_flame_btn = QPushButton("Calculate Equilibrium")
        self.calc_flame_btn.clicked.connect(self.calc_flame)
        flame_layout.addWidget(self.calc_flame_btn)
        
        flame_group.setLayout(flame_layout)
        left_layout.addWidget(flame_group)
        left_layout.addStretch()

        # ================= 右侧结果与筛选区 =================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.flame_result_label = QLabel("等待计算...")
        self.flame_result_label.setWordWrap(True)
        self.flame_result_label.setStyleSheet("font-size: 14px; padding: 10px; background: #f0f0f0; border-radius: 5px;")
        right_layout.addWidget(self.flame_result_label)

        # 阈值筛选控件
        filter_group = QGroupBox("2. Species Threshold Filter")
        filter_layout = QVBoxLayout()
        
        h_filter = QHBoxLayout()
        h_filter.addWidget(QLabel("Min Mole Fraction:"))
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(["1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "1e-8"])
        self.threshold_combo.setCurrentText("1e-5")
        self.threshold_combo.currentTextChanged.connect(self.update_species_table)
        h_filter.addWidget(self.threshold_combo)
        
        self.show_all_check = QCheckBox("Show All Species")
        self.show_all_check.stateChanged.connect(self.update_species_table)
        h_filter.addWidget(self.show_all_check)
        filter_layout.addLayout(h_filter)
        
        filter_group.setLayout(filter_layout)
        right_layout.addWidget(filter_group)

        # 组分表格 (带复选框)
        self.species_table = QTableWidget()
        self.species_table.setColumnCount(4)
        self.species_table.setHorizontalHeaderLabels(["Select", "Species", "Mole Fraction", "ppm"])
        self.species_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.species_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.species_table.setColumnWidth(0, 50)
        right_layout.addWidget(self.species_table)

        # 在右侧布局中，物种表格之后添加一个热力学对比表格
        thermo_group = QGroupBox("热力学状态对比 (反应前 vs 反应后)")
        thermo_layout = QVBoxLayout()
        self.thermo_table = QTableWidget()
        self.thermo_table.setColumnCount(3)
        self.thermo_table.setHorizontalHeaderLabels(["参数", "初始 (反应前)", "平衡 (反应后)"])
        self.thermo_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        thermo_layout.addWidget(self.thermo_table)
        thermo_group.setLayout(thermo_layout)

        # 将 thermo_group 插入到右侧布局中合适的位置，例如放在物种表格和启动按钮之间
        # 如果右侧面板是用 QVBoxLayout，直接 addWidget 即可
        right_layout.addWidget(thermo_group)


        # 启动 HITRAN 按钮
        self.launch_hitran_btn = QPushButton("Launch HITRAN Spectrum Tool (传递 T, P)")
        self.launch_hitran_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; padding: 12px; border-radius: 5px;")
        self.launch_hitran_btn.clicked.connect(self.launch_hitran_gui)
        self.launch_hitran_btn.setEnabled(False)
        right_layout.addWidget(self.launch_hitran_btn)

        # 组装
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([360, 740])
        main_layout.addWidget(splitter)

    def on_oxidizer_changed(self, text):
        if "Air" in text:
            self.oxidizer_edit.setText("O2:1.0 N2:3.76")
        elif "Pure O2" in text:
            self.oxidizer_edit.setText("O2:1.0")
        elif "Oxy-fuel" in text:
            self.oxidizer_edit.setText("O2:1.0 CO2:2.0")
        elif "Custom" in text:
            self.oxidizer_edit.clear()
            self.oxidizer_edit.setFocus()

    def parse_oxidizer(self, ox_str):
        ox_dict = {}
        try:
            for pair in ox_str.split():
                sp, val = pair.split(':')
                ox_dict[sp.strip()] = float(val)
        except ValueError:
            return None
        return ox_dict

    def calc_flame(self):
        fuel = self.fuel_combo.currentText()
        oxidizer = self.parse_oxidizer(self.oxidizer_edit.text())
        if not oxidizer:
            QMessageBox.warning(self, "Error", "Invalid Oxidizer format. Use 'O2:1.0 N2:3.76'.")
            return

        try:
            phi = float(self.phi_edit.text())
            T_init = float(self.t_init_edit.text())
            P_init = float(self.p_init_edit.text()) * ct.one_atm
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid numeric input.")
            return

        eq_method = self.eq_method_combo.currentText()

        success, msg = self.simulator.calculate_equilibrium(
            T_initial=T_init, P_initial=P_init, composition={}, 
            use_equivalence_ratio=True, fuel=fuel, oxidizer=oxidizer, 
            phi=phi, equilibrate_method=eq_method
        )
        
        if success:
            self.current_results = self.simulator.results
            # 【修复】强制转换为 Python 原生 float，防止 PyQt6 报错
            T_eq = float(self.current_results['temperature'])
            p_eq = float(self.current_results['pressure'] / ct.one_atm)
            
            warning_msg = ""
            if T_eq > 3000:
                warning_msg = "<br><font color='red'><b>Warning:</b> T > 3000K (Outside GRI-3.0 valid range)</font>"
                
            self.flame_result_label.setText(
                f"<b>Method:</b> {eq_method} | <b>Fuel:</b> {fuel} | <b>Φ:</b> {phi}<br>"
                f"<b>T_eq:</b> {T_eq:.2f} K | <b>P_eq:</b> {p_eq:.4f} atm"
                f"{warning_msg}"
            )
            self.update_species_table()
            self.launch_hitran_btn.setEnabled(True)
            # 更新热力学对比表格
            self.update_thermo_table()
        else:
            QMessageBox.critical(self, "Failed", msg)

    def update_species_table(self):
        if not self.current_results:
            return

        threshold = float(self.threshold_combo.currentText()) if not self.show_all_check.isChecked() else 0.0
        
        filtered_species = self.simulator.get_species_above_threshold(threshold)
        top_5 = self.simulator.get_top_N_species(n=5)
        
        display_dict = {}
        # 【修复】强制转换为 Python 原生 float
        for sp, frac in top_5.items(): display_dict[sp] = float(frac) 
        for sp, frac in filtered_species.items(): display_dict[sp] = float(frac)

        sorted_species = sorted(display_dict.items(), key=lambda x: x[1], reverse=True)

        self.species_table.setRowCount(len(sorted_species))
        for row, (sp, frac) in enumerate(sorted_species):
            # 【核心修复】将 numpy.bool_ 强制转换为 Python 原生 bool！
            is_checked = bool(frac > 1e-4) 
            
            chk = QCheckBox()
            chk.setChecked(is_checked) # PyQt6 现在能正确接受了
            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.addWidget(chk)
            chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk_layout.setContentsMargins(0,0,0,0)
            self.species_table.setCellWidget(row, 0, chk_widget)
            
            self.species_table.setItem(row, 1, QTableWidgetItem(sp))
            frac_str = f"{frac:.6f}" if frac >= 1e-5 else f"{frac:.2e}"
            self.species_table.setItem(row, 2, QTableWidgetItem(frac_str))
            self.species_table.setItem(row, 3, QTableWidgetItem(f"{frac*1e6:.2f}"))

    def get_selected_species(self):
        selected = {}
        for row in range(self.species_table.rowCount()):
            chk_widget = self.species_table.cellWidget(row, 0)
            if chk_widget:
                chk = chk_widget.findChild(QCheckBox)
                if chk and chk.isChecked():
                    species = self.species_table.item(row, 1).text()
                    frac = float(self.species_table.item(row, 2).text())
                    selected[species] = frac
        return selected

    def update_thermo_table(self):
        """更新热力学状态对比表格"""
        comparison = self.simulator.get_thermodynamic_comparison_data()
        if comparison is None:
            self.thermo_table.setRowCount(0)
            return

        self.thermo_table.setRowCount(len(comparison))
        for row, item in enumerate(comparison):
            self.thermo_table.setItem(row, 0, QTableWidgetItem(item['parameter']))
            self.thermo_table.setItem(row, 1, QTableWidgetItem(item['initial_str']))
            self.thermo_table.setItem(row, 2, QTableWidgetItem(item['final_str']))

    def launch_hitran_gui(self):
        if not self.current_results:
            QMessageBox.warning(self, "Warning", "Please calculate flame equilibrium first.")
            return

        T_eq = float(self.current_results['temperature'])
        p_eq = float(self.current_results['pressure'] / ct.one_atm)
        selected = self.get_selected_species()

        # 使用 HitranSpectrum.ISO 字典构建 name -> mol_id 映射（可缓存在 self 中）
        if not hasattr(self, 'name_to_mol_id'):
            iso_dict = HitranSpectrum.ISO
            self.name_to_mol_id = {}
            for (mol_id, iso_id), vals in iso_dict.items():
                name = vals[4]
                if name not in self.name_to_mol_id:
                    self.name_to_mol_id[name] = mol_id

        # 构建 molecule_dict: {分子名: (par_path, mole_fraction)}
        molecule_dict = {}
        for sp, frac in selected.items():
            mol_id = self.name_to_mol_id.get(sp)
            if mol_id is None:
                continue
            folder_name = f"{mol_id:02d}_{sp}"
            folder_path = os.path.join(self.hitran_db_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            par_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.par')])
            if not par_files:
                continue
            par_path = os.path.join(folder_path, par_files[0])
            molecule_dict[sp] = (par_path, frac)

        if not molecule_dict:
            QMessageBox.warning(self, "Warning", "No matching .par files found for selected species.")
            return

        q_folder = os.path.join(self.hitran_db_dir, 'Q')
        self.hitran_window = HitranGUI(
            initial_T=T_eq,
            initial_P=p_eq,
            q_folder=q_folder,
            cantera_species=selected   # 仍可保留
        )
        self.hitran_window.auto_load_molecules(molecule_dict)

        # 日志
        if hasattr(self.hitran_window, 'info_text'):
            info_msg = f"--- Cantera Equilibrium Results ---\n"
            info_msg += f"T_eq: {T_eq:.2f} K, P_eq: {p_eq:.4f} atm\n"
            info_msg += "Auto-loaded species:\n"
            for sp in molecule_dict:
                info_msg += f"  - {sp}: {molecule_dict[sp][1]*1e6:.1f} ppm\n"
            info_msg += "-----------------------------------\n"
            self.hitran_window.info_text.setPlainText(info_msg)

        self.hitran_window.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FlameSpectrumApp()
    window.show()
    sys.exit(app.exec())

    
