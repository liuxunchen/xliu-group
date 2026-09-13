import sys
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 导入HAPI并读取数据 ==========
hapi_path = r'C:\Users\zhanghongmin\Desktop\hapi-master\hapi-master\hapi'
sys.path.append(hapi_path)
from hapi import *

db_path = r'C:\Users\zhanghongmin\Desktop\CO_simulation'
db_begin(db_path)

# 读取全表数据
molec_ids, iso_ids, nus, sws = getColumns(
    'HITRAN_2073-2074',
    ['molec_id', 'local_iso_id', 'nu', 'sw']
)
total_lines = len(molec_ids)
print(f"✅ 读取到 {total_lines} 条谱线数据")

# 数据类型转换
nus = np.array([float(x) for x in nus])
sws = np.array([float(x) for x in sws])
molec_ids = np.array([int(float(x)) for x in molec_ids])

# ========== 2. 排序 ==========
sorted_idx = np.argsort(sws)[::-1]
sorted_nus = nus[sorted_idx]
sorted_sws = sws[sorted_idx]
sorted_molec_ids = molec_ids[sorted_idx]

# ========== 3. 绘图 ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('2073-2074 cm⁻¹ 气体吸收峰值分析', fontsize=16, fontweight='bold')

# 子图1：波数-强度散点图
ax1.scatter(nus, sws, alpha=0.6, s=10, c='steelblue', label='所有谱线')
for i in range(5):
    ax1.scatter(sorted_nus[i], sorted_sws[i], s=50, c='red', marker='*')
    ax1.annotate(
        moleculeName(sorted_molec_ids[i]),
        (sorted_nus[i], sorted_sws[i]),
        xytext=(5, 5), textcoords='offset points', fontsize=8
    )
ax1.set_xlabel('波数 (cm⁻¹)', fontsize=12)
ax1.set_ylabel('吸收强度 (cm⁻¹/(molec·cm⁻²))', fontsize=12)
ax1.set_title('全区间吸收强度分布', fontsize=14)
ax1.grid(alpha=0.3)

# 子图2：Top10柱状图
top10_molec = [moleculeName(id) for id in sorted_molec_ids[:10]]
top10_sws = sorted_sws[:10]
top10_nus = [f'{x:.2f}' for x in sorted_nus[:10]]

bars = ax2.bar(range(10), top10_sws, color='coral', alpha=0.8)
ax2.set_xticks(range(10))
ax2.set_xticklabels([f'{name}\n({nu})' for name, nu in zip(top10_molec, top10_nus)], fontsize=9)
ax2.set_ylabel('吸收强度 (cm⁻¹/(molec·cm⁻²))', fontsize=12)
ax2.set_title('Top10最强吸收峰值', fontsize=14)
ax2.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x()+bar.get_width()/2., height, f'{height:.2e}', ha='center', va='bottom', fontsize=8)

# ========== 4. 只保存，不show！核心修复 ==========
plt.tight_layout()
save_path = f"{db_path}/2073-2074吸收峰值分析图.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
# 注释掉 plt.show()，避免触发PyCharm后端bug
# plt.show()

print(f"\n✅ 图表已保存至：{save_path}")
print("💡 已跳过 plt.show()，请直接去文件夹打开图片查看！")

# ========== 5. 气体统计（保留） ==========
gas_stats = {}
for mol_id, sw in zip(molec_ids, sws):
    gas_name = moleculeName(mol_id)
    if gas_name not in gas_stats:
        gas_stats[gas_name] = {'count':0, 'total_sw':0.0}
    gas_stats[gas_name]['count'] +=1
    gas_stats[gas_name]['total_sw'] += sw

print("\n" + "="*80)
print("📈 各气体谱线统计")
print("="*80)
for gas_name, stats in sorted(gas_stats.items(), key=lambda x:x[1]['total_sw'], reverse=True):
    avg_sw = stats['total_sw']/stats['count']
    print(f"🔹 {gas_name}：{stats['count']} 条 | 平均强度：{avg_sw:.4e}")