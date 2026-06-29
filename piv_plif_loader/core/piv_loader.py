#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIV数据加载模块 - 提供load_piv函数
支持txt和dat格式，支持bin平均，支持原点平移（可选）
"""

import pathlib
import numpy as np
import pandas as pd
import re

# ============================================================================
# PIV数据加载函数
# ============================================================================

def load_piv(filename, filetype='txt', size=1, x_origin=None, y_origin=None):
    """
    提取PIV数据，支持bin平均和原点平移（可选）
    
    参数:
    -----------
    filename : str or Path
        PIV数据文件路径
    filetype : str
        文件类型 ('txt' 或 'dat')
    size : int
        bin平均大小 (1表示不平均，2表示2x2平均，以此类推)
    x_origin : float or None
        原点x坐标，如果提供则平移坐标系；如果为None则不进行平移
    y_origin : float or None
        原点y坐标，如果提供则平移坐标系；如果为None则不进行平移
    
    返回:
    --------
    dict: 包含PIV数据的字典
        - step: 时间步
        - xnum, ynum: 网格尺寸
        - X, Y: 坐标网格 (如果提供了原点坐标，则已平移至新原点)
        - U, V: 速度分量
        - file_name: 文件名
        - file_path: 文件路径
        - origin_shifted: 布尔值，表示是否进行了原点平移
        - original_x_range: 原始X范围（如果进行了平移）
        - original_y_range: 原始Y范围（如果进行了平移）
    """
    step = np.nan
    size = int(size)
    
    filename = pathlib.Path(filename)
    
    if filetype == 'txt':
        with open(filename, 'r') as lavision:
            header = lavision.readline()
            parts = header.split()
            try:
                step = int(parts[3]) if len(parts) > 3 else np.nan
                xnum = int(parts[4])
                ynum = int(parts[5])
            except (IndexError, ValueError) as e:
                raise ValueError(f"解析txt header失败: {header}") from e
        
        oned = pd.read_csv(filename, decimal=',', sep='\t', skiprows=1, header=None)
        
    elif filetype == 'dat':
        with open(filename, 'r') as f:
            header_lines = [f.readline().strip() for _ in range(3)]
        
        zone_line = header_lines[2]
        match = re.search(r'I=(\d+),\s*J=(\d+)', zone_line)
        if match:
            ynum = int(match.group(1))
            xnum = int(match.group(2))
        else:
            raise ValueError("无法从文件头解析网格尺寸")
        
        oned = pd.read_csv(filename, decimal='.', sep='\\s+', skiprows=3, header=None)
    else:
        raise ValueError(f"不支持的格式: {filetype}")
    
    if oned.shape[1] < 4:
        raise ValueError(f"文件必须至少有4列，实际有 {oned.shape[1]} 列")
    
    X = np.reshape(oned.values[:, 0], (xnum, ynum), 'C').astype('float64')
    Y = np.reshape(oned.values[:, 1], (xnum, ynum), 'C').astype('float64')
    U = np.reshape(oned.values[:, 2], (xnum, ynum), 'C').astype('float64')
    V = np.reshape(oned.values[:, 3], (xnum, ynum), 'C').astype('float64')


    # 保存原始范围（用于信息显示）
    original_x_range = [float(np.min(X)), float(np.max(X))]
    original_y_range = [float(np.min(Y)), float(np.max(Y))]
    
    # 检查Y坐标顺序并翻转（确保Y递增）
    y_sorted = np.sort(Y[:, 0])
    y_original = Y[:, 0]
    
    if not np.allclose(y_sorted, y_original):
        Y = np.flipud(Y)
        X = np.flipud(X)
        U = np.flipud(U)
        V = np.flipud(V)
    
    # Bin平均
    if size != 1:
        x_bins = xnum // size
        y_bins = ynum // size
        
        if x_bins * size != xnum or y_bins * size != ynum:
            print(f"  警告: 网格尺寸 {xnum}x{ynum} 不能被 {size} 整除，将截断")
        
        X = X[:x_bins * size, :y_bins * size].reshape(x_bins, size, y_bins, size).mean(axis=(1, 3))
        Y = Y[:x_bins * size, :y_bins * size].reshape(x_bins, size, y_bins, size).mean(axis=(1, 3))
        U = U[:x_bins * size, :y_bins * size].reshape(x_bins, size, y_bins, size).mean(axis=(1, 3))
        V = V[:x_bins * size, :y_bins * size].reshape(x_bins, size, y_bins, size).mean(axis=(1, 3))
        
        xnum, ynum = X.shape

    
    # 原点平移（仅当提供了x_origin和y_origin时）
    origin_shifted = False
    if x_origin is not None and y_origin is not None:
        X = X - x_origin
        Y = Y - y_origin
        origin_shifted = True

    
    # 转换为float32以节省内存
    X = X.astype('float32')
    Y = Y.astype('float32')
    U = U.astype('float32')
    V = V.astype('float32')


    result = {
        'step': step,
        'xnum': xnum,
        'ynum': ynum,
        'X': X,
        'Y': Y,
        'U': U,
        'V': V,
        'file_name': filename.name,
        'file_path': str(filename),
        'origin_shifted': origin_shifted
    }
    
    # 如果进行了原点平移，保存原始范围信息
    if origin_shifted:
        result['original_x_range'] = original_x_range
        result['original_y_range'] = original_y_range
        result['x_origin_used'] = x_origin
        result['y_origin_used'] = y_origin
    
    return result


def load_piv_batch(file_list, filetype='txt', size=1, x_origin=None, y_origin=None, verbose=True):
    """
    批量加载多个PIV文件
    
    参数:
    -----------
    file_list : list
        文件路径列表
    filetype : str
        文件类型 ('txt' 或 'dat')
    size : int
        bin平均大小
    x_origin, y_origin : float or None
        原点坐标，如果为None则不进行平移
    verbose : bool
        是否打印加载信息
    
    返回:
    --------
    list: 每个文件的PIV数据字典
    """
    results = []
    for file_path in file_list:
        try:
            data = load_piv(file_path, filetype, size, x_origin, y_origin)
            results.append(data)
            if verbose:
                file_name = pathlib.Path(file_path).name
                shift_info = " (已平移)" if data['origin_shifted'] else ""
                print(f"  加载成功: {file_name}{shift_info}")
        except Exception as e:
            if verbose:
                print(f"  加载失败 {file_path}: {e}")
    return results


def get_piv_info(filename, filetype='txt'):
    """
    快速获取PIV文件的基本信息（不加载完整数据）
    
    返回:
    --------
    dict: 包含step, xnum, ynum, 文件大小等信息
    """
    filename = pathlib.Path(filename)
    file_size_kb = filename.stat().st_size / 1024
    
    if filetype == 'txt':
        with open(filename, 'r') as f:
            header = f.readline()
            parts = header.split()
            try:
                step = int(parts[3]) if len(parts) > 3 else np.nan
                xnum = int(parts[4])
                ynum = int(parts[5])
            except (IndexError, ValueError):
                step, xnum, ynum = np.nan, np.nan, np.nan
    elif filetype == 'dat':
        with open(filename, 'r') as f:
            header_lines = [f.readline().strip() for _ in range(3)]
        
        zone_line = header_lines[2]
        match = re.search(r'I=(\d+),\s*J=(\d+)', zone_line)
        if match:
            ynum = int(match.group(1))
            xnum = int(match.group(2))
        else:
            xnum, ynum = np.nan, np.nan
        step = np.nan
    else:
        raise ValueError(f"不支持的格式: {filetype}")
    
    return {
        'filename': filename.name,
        'file_size_kb': file_size_kb,
        'step': step,
        'xnum': xnum,
        'ynum': ynum,
        'total_points': xnum * ynum if not np.isnan(xnum) else np.nan
    }


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    """
    使用示例:
    
    # 不进行原点平移（使用原始坐标）
    data = load_piv('B00100.txt', filetype='txt', size=1)
    
    # 进行原点平移
    data = load_piv('B00100.txt', filetype='txt', size=1, x_origin=4.0, y_origin=-27.0)
    
    # 批量加载
    files = ['B00100.txt', 'B00101.txt']
    results = load_piv_batch(files, filetype='txt', size=2, x_origin=4.0, y_origin=-27.0)
    """
    print("PIV数据加载模块")
    print("="*50)
    print("此模块提供以下函数:")
    print("  - load_piv: 加载单个PIV文件")
    print("  - load_piv_batch: 批量加载多个PIV文件")
    print("  - get_piv_info: 获取PIV文件基本信息")
    print("="*50)
    print("\n使用示例:")
    print('''
    from piv_loader import load_piv
    
    # 不进行原点平移
    data = load_piv('B00100.txt', filetype='txt', size=1)
    print(f"原始X范围: [{np.min(data['X']):.1f}, {np.max(data['X']):.1f}]")
    
    # 进行原点平移
    data_shifted = load_piv('B00100.txt', filetype='txt', size=1, 
                            x_origin=4.0, y_origin=-27.0)
    print(f"平移后X范围: [{np.min(data_shifted['X']):.1f}, {np.max(data_shifted['X']):.1f}]")
    ''')
