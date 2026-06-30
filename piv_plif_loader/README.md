# PIV/PLIF Liutex Flow Field Analyzer

跨平台、模块化的流场后处理 GUI 工具，集成 PIV/PLIF 数据加载、Liutex 涡识别、多物理量可视化与频谱分析。

## 功能

- **PIV 数据加载**：支持 DaVis 标准 `.txt`（逗号小数/制表符分隔）、TECPLOT 风格 `.dat` 以及 DaVis 原始向量导出格式
- **PLIF 数据加载**：读取 TIFF 图像序列，支持用户指定标定参数（像素物理尺寸），提供边缘保持滤波、阈值掩膜、模糊增强等预处理
- **Liutex 涡识别**：严格实现二维 Liutex 分解，输出纯旋转强度 R、剪切贡献 S 与局部旋转角速度 λ_ci
- **多物理量可视化**：合速度、涡量、散度、梯度、Liutex，可选叠加速度矢量（quiver）与流线（streamline）
- **PLIF 叠加**：将处理后的 PLIF 标量场以半透明形式覆盖在速度场之上
- **实时动画**：播放/暂停，滑块拖拽浏览序列帧
- **单点 FFT**：在流场上选点，提取该点时域序列并绘制幅值谱
- **数据导出**：当前标量场保存为 `.npy` 文件

## 快速开始

```bash
pip install -r requirements.txt
python app.py
```

## 使用方法

1. 点击 **加载 PIV 序列** 选择 DaVis 导出的 `.txt` 或 `.dat` 文件（支持多选）
2. （可选）点击 **加载 PLIF 序列** 选择 TIFF 图像；会弹出标定对话框输入像素物理尺寸
3. 在下拉框选择要显示的物理量（涡量、散度、Liutex 等）
4. 勾选 quiver / streamline 叠加矢量或流线
5. 勾选 **叠加 PLIF** 显示处理后的 PLIF 标量场
6. 使用工具栏缩放/平移图像
7. 点击 **在流场上选点** 后在图像上单击，查看该点 FFT 频谱

## 项目结构

```
piv_plif_loader/
├── app.py                      # 程序入口
├── core/
│   ├── piv_loader.py           # PIV 数据加载
│   ├── plif_loader.py          # PLIF TIFF 数据加载与增强
│   ├── liutex.py               # Liutex 涡识别计算
│   ├── field_calculations.py   # 其他物理量计算
│   └── fft_analyzer.py         # FFT 频谱分析
├── gui/
│   ├── __init__.py
│   ├── main_window.py          # 主窗口（信号/槽整合）
│   ├── controls.py             # 控制面板
│   ├── canvas.py               # Matplotlib 画布
│   └── dialogs.py              # 对话框（FFT, PLIF 标定等）
├── data/
│   └── PIV_example/            # 示例 PIV 数据
├── requirements.txt
└── README.md
```

## 依赖

- Python ≥ 3.9
- PyQt6
- NumPy, Pandas, SciPy
- Matplotlib
- OpenCV (用于 TIFF 读取和图像增强)
- 可选：imageio / tifffile（OpenCV 无法读取某些 TIFF 时的备用）

## 设计原则

核心算法模块（`core/`）与 GUI 完全分离，不依赖 Qt，可独立测试与嵌入批处理脚本。GUI 层通过信号/槽机制响应控件变化，确保代码可维护性与可扩展性。

## License

MIT
