# core/flame_simulator.py
import os
import cantera as ct

class GasCompositionSimulator:
    """气体组分模拟核心类 (Cantera 化学平衡)"""

    def __init__(self):
        self.gas = None
        self.initial_state = None
        self.final_state = None
        self.results = None
        self.initial_conditions = {}
        self.mechanism_files = self.find_mechanism_files()

    def find_mechanism_files(self):
        mechanism_files = []
        for file in os.listdir('.'):
            if file.endswith(('.yaml', '.cti')):
                mechanism_files.append(file)
        if not mechanism_files:
            mechanism_files = ['gri30.yaml']
        return mechanism_files

    def initialize_gas(self, mechanism_file='gri30.yaml'):
        try:
            self.gas = ct.Solution(mechanism_file)
            return True, f"成功加载反应机理: {mechanism_file}"
        except Exception as e:
            return False, f"加载反应机理失败: {e}"

    def get_thermodynamic_state(self, gas):
        """获取热力学状态（每摩尔）"""
        return {
            'pressure': gas.P,                             # Pa
            'temperature': gas.T,                          # K
            'volume': gas.volume_mole * 1000,              # m³/mol -> L/mol
            'internal_energy': int(gas.int_energy_mole / 1000),  # J/mol -> kJ/mol
            'enthalpy': int(gas.enthalpy_mole / 1000),     # J/mol -> kJ/mol
            'entropy': int(gas.entropy_mole / 1000),       # J/mol/K -> kJ/mol/K
            'gibbs': int(gas.gibbs_mole / 1000),           # J/mol -> kJ/mol
            'cp': int(gas.cp_mole / 1000),                 # J/mol/K -> kJ/mol/K
            'cv': int(gas.cv_mole / 1000),                 # J/mol/K -> kJ/mol/K
            'density': gas.density_mole,                   # kmol/m³
            'mean_molecular_weight': gas.mean_molecular_weight  # kg/kmol
        }

    def calculate_equilibrium(self, T_initial, P_initial, composition,
                              use_equivalence_ratio=False,
                              fuel=None, oxidizer=None, phi=1.0,
                              equilibrate_method='HP'):
        try:
            if use_equivalence_ratio and fuel and oxidizer:
                self.gas.TP = T_initial, P_initial
                self.gas.set_equivalence_ratio(phi, fuel, oxidizer)
            else:
                self.gas.TPX = T_initial, P_initial, composition

            # 初始状态（反应前）
            self.initial_state = self.get_thermodynamic_state(self.gas)

            # 执行化学平衡
            self.gas.equilibrate(equilibrate_method)

            # 最终状态（反应后）
            self.final_state = self.get_thermodynamic_state(self.gas)

            # 保存结果
            final_composition = dict(zip(self.gas.species_names, self.gas.X))
            self.results = {
                'temperature': self.gas.T,
                'pressure': self.gas.P,
                'mole_fractions': final_composition,
                'species_names': self.gas.species_names
            }
            return True, "平衡计算成功"
        except Exception as e:
            return False, f"平衡计算失败: {e}"

    def get_thermodynamic_comparison_data(self):
        """返回初始状态与平衡状态的对比数据（用于表格显示）"""
        if not self.initial_state or not self.final_state:
            return None
        parameters = [
            ('压力 (Pa)', 'pressure', '{:>12.0f}'),
            ('温度 (K)', 'temperature', '{:>12.0f}'),
            ('体积 (L/mol)', 'volume', '{:>12.6f}'),
            ('内能 (kJ/mol)', 'internal_energy', '{:>12.0f}'),
            ('焓 (kJ/mol)', 'enthalpy', '{:>12.0f}'),
            ('熵 (kJ/mol/K)', 'entropy', '{:>12.0f}')
        ]
        data = []
        for name, key, fmt in parameters:
            data.append({
                'parameter': name,
                'initial_str': fmt.format(self.initial_state[key]),
                'final_str': fmt.format(self.final_state[key])
            })
        return data

    def get_species_above_threshold(self, threshold=1e-6):
        """获取高于指定阈值的物种，并按浓度降序排列"""
        if self.results is None:
            return {}
        mole_fractions = self.results['mole_fractions']
        filtered = {k: v for k, v in mole_fractions.items() if v >= threshold}
        return dict(sorted(filtered.items(), key=lambda item: item[1], reverse=True))

    def get_top_N_species(self, n=10):
        """获取浓度最高的前 N 个物种（无视阈值）"""
        if self.results is None:
            return {}
        mole_fractions = self.results['mole_fractions']
        sorted_all = dict(sorted(mole_fractions.items(), key=lambda item: item[1], reverse=True))
        return {k: v for k, v in list(sorted_all.items())[:n] if v > 0}