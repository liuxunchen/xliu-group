#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据加载模块 - 支持 PIV (txt/dat/davis) 和 PLIF (txt)
功能：读取、bin平均、原点平移、坐标对齐、PLIF增强
"""

import pathlib
import numpy as np
import pandas as pd
import re
from typing import Optional, Union, Tuple, Dict, Any, List


# ============================================================================
# 内部工具函数
# ============================================================================

def _read_piv_header(filename: pathlib.Path, filetype: str) -> Tuple[int, int, int]:
    """解析 PIV 文件头 (txt/dat)，返回 step, xnum, ynum"""
    if filetype == 'txt':
        with open(filename, 'r') as f:
            header = f.readline()
            parts = header.split()
            try:
                step = int(parts[3]) if len(parts) > 3 else np.nan
                xnum = int(parts[4])
                ynum = int(parts[5])
            except (IndexError, ValueError) as e:
                raise ValueError(f"无法解析txt文件头: {header}") from e
        return step, xnum, ynum

    elif filetype == 'dat':
        with open(filename, 'r') as f:
            header_lines = [f.readline().strip() for _ in range(3)]
        zone_line = header_lines[2]
        match = re.search(r'I=(\d+),\s*J=(\d+)', zone_line)
        if match:
            ynum = int(match.group(1))  # I = 行数 (y方向)
            xnum = int(match.group(2))  # J = 列数 (x方向)
        else:
            raise ValueError("无法从dat文件头解析网格尺寸")
        return np.nan, xnum, ynum

    else:
        raise ValueError(f"不支持的PIV文件类型: {filetype}")


def _parse_davis_header(filename: pathlib.Path) -> Tuple[int, int]:
    """解析 DaVis 向量导出格式，返回 (xnum, ynum) （无 step）"""
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
        if not first_line.startswith('#DaVis'):
            raise ValueError("不是 DaVis 格式")
        parts = first_line.split()
        # 典型格式：#DaVis 8.4.0 2D-vector 3 220 257 "" "mm" "" "mm" "velocity" "m/s"
        try:
            idx = parts.index('2D-vector')
            xnum = int(parts[idx + 2])
            ynum = int(parts[idx + 3])
        except (ValueError, IndexError):
            # 备用策略：找所有纯数字，取最后两个
            nums = [int(p) for p in parts if p.isdigit()]
            if len(nums) >= 2:
                xnum, ynum = nums[-2], nums[-1]
            else:
                raise ValueError("无法从 DaVis 头解析网格尺寸")
    return xnum, ynum


def _apply_bin_average(arr: np.ndarray, bin_size: int) -> np.ndarray:
    """对2D数组进行bin平均（下采样），仅保留完整bin区域"""
    if bin_size == 1:
        return arr
    ny, nx = arr.shape
    new_ny = ny // bin_size
    new_nx = nx // bin_size
    arr = arr[:new_ny * bin_size, :new_nx * bin_size]
    reshaped = arr.reshape(new_ny, bin_size, new_nx, bin_size)
    return reshaped.mean(axis=(1, 3))


def _postprocess_piv(X: np.ndarray, Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                     xnum: int, ynum: int, step: float,
                     filename: pathlib.Path, size: int,
                     x_origin: Optional[float], y_origin: Optional[float]
                     ) -> Dict[str, Any]:
    """
    通用后处理：确保Y轴递增、bin平均、转置为(ny, nx)、原点平移、类型转换。
    输入 X, Y, U, V 形状为 (xnum, ynum) 的二维数组。
    """
    original_x_range = (float(np.min(X)), float(np.max(X)))
    original_y_range = (float(np.min(Y)), float(np.max(Y)))

    # 1. 确保 Y 轴物理坐标递增（若递减则翻转）
    if Y[0, 0] > Y[-1, 0]:
        X = np.flipud(X)
        Y = np.flipud(Y)
        U = np.flipud(U)
        V = np.flipud(V)

    # 2. Bin 平均
    X = _apply_bin_average(X, size)
    Y = _apply_bin_average(Y, size)
    U = _apply_bin_average(U, size)
    V = _apply_bin_average(V, size)

    # 3. 转置为 (ynum, xnum) 以符合 numpy 行/列习惯
    #    此时 X 形状为 (xnum//size, ynum//size) 或已翻转
    if X.shape[0] == xnum // size and X.shape[1] == ynum // size:
        X = X.T
        Y = Y.T
        U = U.T
        V = V.T
        xnum_new, ynum_new = X.shape[1], X.shape[0]  # nx, ny
    else:
        xnum_new, ynum_new = X.shape[1], X.shape[0]

    # 4. 原点平移（可选）
    origin_shifted = False
    if x_origin is not None and y_origin is not None:
        X = X - x_origin
        Y = Y - y_origin
        origin_shifted = True

    # 5. 转换为 float32 节省内存
    X = X.astype('float32')
    Y = Y.astype('float32')
    U = U.astype('float32')
    V = V.astype('float32')

    result = {
        'step': step,
        'xnum': xnum_new,
        'ynum': ynum_new,
        'X': X,
        'Y': Y,
        'U': U,
        'V': V,
        'file_name': filename.name,
        'file_path': str(filename),
        'origin_shifted': origin_shifted,
        'original_x_range': original_x_range,
        'original_y_range': original_y_range
    }
    if origin_shifted:
        result['x_origin_used'] = x_origin
        result['y_origin_used'] = y_origin
    return result


# ============================================================================
# 公共接口
# ============================================================================

def load_piv(filename: Union[str, pathlib.Path],
             filetype: str = 'txt',
             size: int = 1,
             x_origin: Optional[float] = None,
             y_origin: Optional[float] = None) -> Dict[str, Any]:
    """
    加载单帧 PIV 数据，支持 bin 平均和可选原点平移。

    Parameters
    ----------
    filename : str or Path
        文件路径。
    filetype : str
        'txt'   - DaVis 标准文本（制表符分隔，逗号小数）
        'dat'   - TECPLOT 风格（空格分隔，点号小数）
        'davis' - DaVis 原始向量导出（#DaVis 头，空格分隔，点号小数）
    size : int
        bin 平均大小，1 表示不平均。
    x_origin, y_origin : float or None
        若提供，将坐标系平移至此点（X - x_origin, Y - y_origin）。

    Returns
    -------
    dict : 包含 X, Y, U, V, step, 网格尺寸等
    """
    filename = pathlib.Path(filename)
    size = int(size)

    # ---------- DaVis 原始向量格式 ----------
    if filetype == 'davis':
        xnum, ynum = _parse_davis_header(filename)
        step = np.nan
        df = pd.read_csv(filename, comment='#', sep=r'\s+', header=None)
        X = np.reshape(df.values[:, 0], (xnum, ynum), 'C').astype('float64')
        Y = np.reshape(df.values[:, 1], (xnum, ynum), 'C').astype('float64')
        U = np.reshape(df.values[:, 2], (xnum, ynum), 'C').astype('float64')
        V = np.reshape(df.values[:, 3], (xnum, ynum), 'C').astype('float64')
        return _postprocess_piv(X, Y, U, V, xnum, ynum, step, filename, size,
                                x_origin, y_origin)

    # ---------- 标准 txt / dat ----------
    step, xnum, ynum = _read_piv_header(filename, filetype)

    if filetype == 'txt':
        df = pd.read_csv(filename, decimal=',', sep='\t', skiprows=1, header=None)
    elif filetype == 'dat':
        df = pd.read_csv(filename, decimal='.', sep='\\s+', skiprows=3, header=None)
    else:
        raise ValueError(f"不支持的 filetype: {filetype}")

    X = np.reshape(df.values[:, 0], (xnum, ynum), 'C').astype('float64')
    Y = np.reshape(df.values[:, 1], (xnum, ynum), 'C').astype('float64')
    U = np.reshape(df.values[:, 2], (xnum, ynum), 'C').astype('float64')
    V = np.reshape(df.values[:, 3], (xnum, ynum), 'C').astype('float64')

    return _postprocess_piv(X, Y, U, V, xnum, ynum, step, filename, size,
                            x_origin, y_origin)


def load_plif(filename: Union[str, pathlib.Path],
              size: int = 1) -> Dict[str, Any]:
    """
    加载 PLIF 标量场数据 (txt 格式)。

    文件头示例：
        x_num y_num delta_x x0 delta_y y0 ...
    数据部分为制表符分隔的浮点数，每行一个网格点的标量值。

    Parameters
    ----------
    filename : str or Path
        PLIF 文件路径。
    size : int
        bin 平均大小，1 表示不平均。

    Returns
    -------
    dict : 包含 'x_num', 'y_num', 'X', 'Y' (1D 坐标), 'scalar' (2D 数组)
    """
    filename = pathlib.Path(filename)
    size = int(size)

    with open(filename, 'r') as f:
        header = f.readline().split()
        x_num = int(header[3])
        y_num = int(header[4])
        delta_x = float(header[6])
        x0 = float(header[7])
        delta_y = float(header[10])
        y0 = float(header[11])

    # 读取数据（制表符分隔，逗号小数）
    df = pd.read_csv(filename, decimal=',', sep='\t', skiprows=1, header=None)
    scalar = df.values
    # 假设标量只有一列，取第一列
    if scalar.ndim == 2:
        scalar = scalar[:, 0]
    scalar_2d = np.reshape(scalar, (x_num, y_num), 'C').astype('float64')

    # 若 delta_y < 0，翻转 Y 轴使其递增
    if delta_y < 0:
        y0 = y0 + (y_num - 1) * delta_y  # 新的最小 y
        delta_y = -delta_y
        scalar_2d = np.flipud(scalar_2d)

    # Bin 平均
    scalar_2d = _apply_bin_average(scalar_2d, size)
    new_y_num, new_x_num = scalar_2d.shape  # 注意 _apply_bin_average 保持原维度顺序

    # 重新生成坐标（使用原始间距乘以 bin 大小）
    X_new = np.arange(x0, x0 + new_x_num * delta_x * size, delta_x * size)
    Y_new = np.arange(y0, y0 + new_y_num * delta_y * size, delta_y * size)

    # 转置为 (y_num, x_num) 习惯
    scalar_2d = scalar_2d.T  # 现在形状是 (y_num, x_num)

    return {
        'x_num': new_x_num,
        'y_num': new_y_num,
        'X': X_new.astype('float32'),
        'Y': Y_new.astype('float32'),
        'scalar': scalar_2d.astype('float32'),
        'file_name': filename.name,
        'file_path': str(filename)
    }


def load_piv_batch(file_list: List[Union[str, pathlib.Path]],
                   filetype: str = 'txt',
                   size: int = 1,
                   x_origin: Optional[float] = None,
                   y_origin: Optional[float] = None,
                   verbose: bool = True) -> List[Dict[str, Any]]:
    """批量加载多个 PIV 文件"""
    results = []
    for f in file_list:
        try:
            data = load_piv(f, filetype, size, x_origin, y_origin)
            results.append(data)
            if verbose:
                shift_info = " (已平移)" if data['origin_shifted'] else ""
                print(f"  已加载: {data['file_name']}{shift_info}")
        except Exception as e:
            if verbose:
                print(f"  加载失败 {f}: {e}")
    return results


def load_plif_batch(file_list: List[Union[str, pathlib.Path]],
                    size: int = 1,
                    verbose: bool = True) -> List[Dict[str, Any]]:
    """批量加载 PLIF 文件"""
    results = []
    for f in file_list:
        try:
            data = load_plif(f, size)
            results.append(data)
            if verbose:
                print(f"  已加载 PLIF: {data['file_name']}")
        except Exception as e:
            if verbose:
                print(f"  加载失败 {f}: {e}")
    return results



# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    # 示例用法：需替换为实际文件路径
    test_piv_file = "../data/PIV_example/B00001.txt"
    test_plif_file = "test/B00001_plif.txt"

    if pathlib.Path(test_piv_file).exists():
        print("测试 PIV 加载（txt 格式）")
        piv = load_piv(test_piv_file, filetype='txt', size=1)
        print("  PIV keys:", list(piv.keys()))
        print("  X shape:", piv['X'].shape, "U range:", piv['U'].min(), piv['U'].max())

        # 也可测试 davis 格式
        # piv2 = load_piv(test_piv_file, filetype='davis', size=1)

    if pathlib.Path(test_plif_file).exists():
        print("测试 PLIF 加载")
        plif = load_plif(test_plif_file, size=1)
        print("  PLIF keys:", list(plif.keys()))
        print("  scalar shape:", plif['scalar'].shape)

        # 对齐示例（需有对应 piv 数据）
        # aligned = align_plif_to_piv(plif, piv)
