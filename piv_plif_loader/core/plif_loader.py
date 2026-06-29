#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLIF 数据加载模块 - 读取 TIFF 标量图像
支持 bin 平均、可选物理坐标
"""

import pathlib
import numpy as np
import cv2
from typing import Optional, Union, Dict, Any, List


def load_plif(filename: Union[str, pathlib.Path],
              size: int = 1,
              x0: float = 0.0,
              y0: float = 0.0,
              dx: float = 1.0,
              dy: float = 1.0) -> Dict[str, Any]:
    """
    读取单张 TIFF 格式的 PLIF 图像。

    Parameters
    ----------
    filename : str or Path
        TIFF 文件路径。
    size : int
        bin 平均大小 (1 表示不平均)。
    x0, y0 : float
        图像左上角（最小 X, 最小 Y）的物理坐标。
        注意：图像读取后 Y 轴会被翻转，使得 y0 对应物理最小 Y，Y 向上递增。
    dx, dy : float
        单个像素在 x 和 y 方向的物理尺寸（必须为正）。

    Returns
    -------
    dict :
        'x_num', 'y_num' : 最终网格尺寸（列数 × 行数）
        'X', 'Y'         : 1D 物理坐标数组（递增）
        'scalar'         : 2D 标量场，形状 (y_num, x_num)，float32
        'file_name', 'file_path'
    """
    filename = pathlib.Path(filename)
    size = int(size)
    if dx <= 0 or dy <= 0:
        raise ValueError("dx 和 dy 必须为正值")

    # 读取图像（保持原始深度，例如 16-bit）
    img = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"无法读取图像文件: {filename}")
    if img.ndim == 3:
        # 若为彩色/多通道，取第一通道
        img = img[:, :, 0]

    # 图像通常 y 轴向下，为与物理坐标系 Y 向上一致，上下翻转
    img = np.flipud(img)
    ny, nx = img.shape

    # Bin 平均
    scalar = _apply_bin_average(img.astype('float64'), size)
    ny_new, nx_new = scalar.shape

    # 重新计算物理坐标（注意翻转后 y0 仍为最小 Y，dy 为正）
    X = np.arange(x0, x0 + nx_new * dx * size, dx * size, dtype='float32')
    Y = np.arange(y0, y0 + ny_new * dy * size, dy * size, dtype='float32')

    return {
        'x_num': nx_new,
        'y_num': ny_new,
        'X': X,
        'Y': Y,
        'scalar': scalar.astype('float32'),
        'file_name': filename.name,
        'file_path': str(filename)
    }


def load_plif_batch(file_list: List[Union[str, pathlib.Path]],
                    size: int = 1,
                    x0: float = 0.0,
                    y0: float = 0.0,
                    dx: float = 1.0,
                    dy: float = 1.0,
                    verbose: bool = True) -> List[Dict[str, Any]]:
    """
    批量加载多个 PLIF 文件。

    Parameters
    ----------
    file_list : list
        文件路径列表。
    size, x0, y0, dx, dy : 同 load_plif。
    verbose : bool
        是否打印加载信息。

    Returns
    -------
    list of dict : 每个文件的 PLIF 数据字典
    """
    results = []
    for f in file_list:
        try:
            data = load_plif(f, size, x0, y0, dx, dy)
            results.append(data)
            if verbose:
                print(f"  已加载 PLIF: {data['file_name']}")
        except Exception as e:
            if verbose:
                print(f"  加载失败 {f}: {e}")
    return results


def _apply_bin_average(arr: np.ndarray, bin_size: int) -> np.ndarray:
    """2D 数组 bin 平均（下采样）"""
    if bin_size == 1:
        return arr
    ny, nx = arr.shape
    ny_new = ny // bin_size
    nx_new = nx // bin_size
    arr = arr[:ny_new * bin_size, :nx_new * bin_size]
    reshaped = arr.reshape(ny_new, bin_size, nx_new, bin_size)
    return reshaped.mean(axis=(1, 3))

def align_plif_to_piv(plif_data: Dict[str, Any],
                      piv_data: Dict[str, Any]) -> np.ndarray:
    """
    将 PLIF 标量场线性插值到 PIV 网格上，以便叠加显示。

    plif_data 包含 1D 坐标 X, Y，以及 2D scalar；
    piv_data 包含 2D 网格 X, Y（形状 ny×nx）。
    返回插值后的标量场，形状与 piv X 相同。
    """
    from scipy.interpolate import griddata

    Xp, Yp = np.meshgrid(plif_data['X'], plif_data['Y'])
    points = np.column_stack((Xp.ravel(), Yp.ravel()))
    values = plif_data['scalar'].ravel()

    Xpiv = piv_data['X']  # 2D, (ny, nx)
    Ypiv = piv_data['Y']

    aligned = griddata(points, values, (Xpiv, Ypiv), method='linear',
                       fill_value=np.nan)
    return aligned.astype('float32')


def enhance_plif(scalar_raw: np.ndarray,
                 sigma_s: int = 10,
                 sigma_r: float = 10.0,
                 thres_hold: int = 30,
                 blur_size: int = 15) -> np.ndarray:
    """
    对 PLIF 标量场进行边缘保持滤波、阈值掩膜和模糊增强，
    用于更好的可视化叠加。

    参数
    ----
    scalar_raw : 2D numpy 数组 (ny, nx)，浮点型
    sigma_s, sigma_r : 边缘保持滤波参数
    thres_hold : 二值化阈值（0-255）
    blur_size : 模糊核大小

    返回
    -------
    enhanced : 2D numpy 数组 (ny, nx)，uint8 范围内的浮点值
    """
    import cv2

    # 归一化到 0-255
    norm = cv2.normalize(scalar_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 三通道供边缘保持滤波使用
    src = cv2.merge([norm, norm, norm])
    filtered = cv2.edgePreservingFilter(src, flags=2, sigma_s=sigma_s, sigma_r=sigma_r)
    gray = filtered[:, :, 0]

    # 阈值掩膜
    _, mask = cv2.threshold(gray, thres_hold, 1, cv2.THRESH_BINARY)

    # 模糊与归一化
    blurred = cv2.blur(norm, (blur_size, blur_size))
    blurred_norm = cv2.normalize(blurred, None, 1, 100, cv2.NORM_MINMAX)

    # 应用掩膜
    enhanced = (blurred_norm * mask).astype(np.float32)
    return enhanced
