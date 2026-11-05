#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLIF 数据加载模块 - 读取 TIFF 标量图像
支持 bin 平均、可选物理坐标、多格式自动回退、
Photron FASTCAM .cihx 元数据解析，原始文件夹批量加载。
"""

import xml.etree.ElementTree as ET
import glob
import os
import pathlib
import numpy as np
from typing import Optional
from typing import Union, Dict, Any, List


def _read_tiff(filename: str) -> np.ndarray:
    """读取 TIFF 文件，tifffile 优先 (避免 cv2 卡死)。"""
    try:
        import tifffile
        img = tifffile.imread(str(filename))
        if img is not None:
            return np.asarray(img)
    except Exception:
        pass
    try:
        import imageio.v3 as iio
        img = iio.imread(str(filename))
        if img is not None:
            return np.asarray(img)
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(str(filename)) as im:
            img = np.array(im)
            if img is not None:
                return img
    except Exception:
        pass
    try:
        import cv2
        img = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass
    raise IOError(f"无法读取图像文件: {filename}")
def parse_cihx(cihx_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """
    解析 Photron FASTCAM .cihx 文件，提取相机参数和图像元数据。

    .cihx 文件是 XML 格式，但前面有一段二进制头（含 "CIHX" 签名），
    XML 内容从 ``<?xml`` 开始。此函数自动跳过二进制前缀。

    Returns
    -------
    dict : 包含 keys:
        - width, height : 图像分辨率
        - bit_depth : 存储位深 (16)
        - effective_depth : 有效位深 (e.g. 12)
        - effective_side : 'Higher' 或 'Lower'
        - record_rate : 录制帧率 (fps)
        - shutter_speed : 快门速度 (分母, e.g. 160000 -> 1/160000 s)
        - total_frame : 总帧数
        - date, time : 录制日期时间
        - camera_name : 相机名称
        - links : 关联的分区文件夹列表
    """
    cihx_path = pathlib.Path(cihx_path)
    raw = cihx_path.read_bytes()

    # 跳过二进制头：找到 "<?xml" 的位置
    xml_start = raw.find(b'<?xml')
    if xml_start >= 0:
        xml_bytes = raw[xml_start:]
    else:
        xml_bytes = raw

    root = ET.fromstring(xml_bytes.decode('utf-8'))

    def _text(path: str, default=None):
        el = root.find(path)
        return el.text if el is not None else default

    def _int(path: str, default=0):
        t = _text(path)
        return int(t) if t else default

    img_data = root.find('imageDataInfo')
    if img_data is not None:
        width = _int('imageDataInfo/resolution/width', 0)
        height = _int('imageDataInfo/resolution/height', 0)
        bit_depth = _int('imageDataInfo/colorInfo/bit', 0)
        effective_depth = _int('imageDataInfo/effectiveBit/depth', 0)
        effective_side = _text('imageDataInfo/effectiveBit/side', '')
    else:
        width = height = bit_depth = effective_depth = 0
        effective_side = ''

    total_frame = _int('frameInfo/totalFrame', 0)
    recorded_frame = _int('frameInfo/recordedFrame', 0)
    record_rate = _int('recordInfo/recordRate', 0)
    shutter_speed = _int('recordInfo/shutterSpeed', 0)
    date = _text('fileInfo/date', '')
    time = _text('fileInfo/time', '')
    camera_name = _text('basicInfo/cameraName', '')

    links = []
    links_el = root.find('imageFileInfo/links')
    if links_el is not None:
        for link_el in links_el.findall('link'):
            if link_el.text:
                links.append(link_el.text)

    return {
        'width': width,
        'height': height,
        'bit_depth': bit_depth,
        'effective_depth': effective_depth,
        'effective_side': effective_side,
        'record_rate': record_rate,
        'shutter_speed': shutter_speed,
        'shutter_exposure': 1.0 / shutter_speed if shutter_speed > 0 else None,
        'total_frame': total_frame,
        'recorded_frame': recorded_frame,
        'date': date,
        'time': time,
        'camera_name': camera_name,
        'cihx_path': str(cihx_path),
        'links': links,
    }


def load_plif_raw_folder(folder: Union[str, pathlib.Path],
                         size: int = 1,
                         bit_shift: Optional[int] = None,
                         x0: float = 0.0,
                         y0: float = 0.0,
                         dx: float = 1.0,
                         dy: float = 1.0,
                         max_frames: Optional[int] = None,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    加载 Photron FASTCAM 原始 PLIF 数据文件夹。

    自动查找文件夹中的 .cihx 文件和 .tif/.tiff 图像序列。

    Parameters
    ----------
    folder : str or Path
        包含 PLIF 原始 TIFF 和一个 .cihx 文件的文件夹路径。
    size : int
        bin 平均大小。
    bit_shift : int or None
        若 Photron 数据有效位存储在"高位"，可指定右移位数。
        None 时自动从 cihx 计算 (bit_depth - effective_depth)。
    x0, y0, dx, dy : 物理标定参数。
    max_frames : int or None
        最大加载帧数，None = 全部。

    Returns
    -------
    dict : {'frames': [...], 'metadata': {...}, 'folder': str}
    """
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        raise ValueError(f"路径不是文件夹: {folder}")

    cihx_files = list(folder.glob('*.cihx'))
    metadata = None
    shift = 0
    if cihx_files:
        metadata = parse_cihx(cihx_files[0])
        if verbose:
            print(f"  解析 cihx: {cihx_files[0].name}")
            print(f"    相机: {metadata['camera_name']}, "
                  f"分辨率: {metadata['width']}x{metadata['height']}, "
                  f"帧率: {metadata['record_rate']} fps")
        if bit_shift is not None:
            shift = bit_shift
        elif metadata['effective_side'].lower() == 'higher' and metadata['effective_depth'] > 0:
            shift = metadata['bit_depth'] - metadata['effective_depth']
            if verbose:
                print(f"    有效位 {metadata['effective_depth']}bit 高位存储, 自动右移 {shift} 位")

    tiff_files = sorted(folder.glob('*.tif')) + sorted(folder.glob('*.tiff'))
    if not tiff_files:
        raise FileNotFoundError(f"文件夹中没有找到 TIFF 文件: {folder}")

    if max_frames is not None:
        tiff_files = tiff_files[:max_frames]

    if verbose:
        print(f"  找到 {len(tiff_files)} 个 TIFF 文件")

    frames = load_plif_batch(
        tiff_files, size=size, x0=x0, y0=y0, dx=dx, dy=dy,
        bit_shift=shift, verbose=verbose
    )

    return {'frames': frames, 'metadata': metadata, 'folder': str(folder)}


def load_plif(filename: Union[str, pathlib.Path],
              size: int = 1,
              bit_shift: int = 0,
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
    bit_shift : int
        右移位数，用于 Photron 高位存储的原始数据。0 表示不移位。
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
    img = _read_tiff(str(filename))
    if img.ndim == 3:
        # 若为彩色/多通道，取亮度（加权平均）
        if img.shape[2] >= 3:
            img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            img = img[:, :, 0]

    # Photron 高位存储：右移 bit_shift 位获得有效像素值
    if bit_shift > 0:
        img = img >> bit_shift

    # 确保为浮点类型
    if np.issubdtype(img.dtype, np.integer):
        img = img.astype('float64')

    # 图像通常 y 轴向下，为与物理坐标系 Y 向上一致，上下翻转
    img = np.flipud(img)
    ny, nx = img.shape

    # Bin 平均
    scalar = _apply_bin_average(img, size)
    ny_new, nx_new = scalar.shape

    # 重新计算物理坐标（注意翻转后 y0 仍为最小 Y，dy 为正）
    X = np.arange(x0, x0 + nx_new * dx * size, dx * size, dtype='float32')
    Y = np.arange(y0, y0 + ny_new * dy * size, dy * size, dtype='float32')

    # 确保坐标数组长度匹配（float32 累加误差可能导致多出元素）
    if len(X) > nx_new:
        X = X[:nx_new]
    if len(Y) > ny_new:
        Y = Y[:ny_new]

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
                    bit_shift: int = 0,
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
    size, bit_shift, x0, y0, dx, dy : 同 load_plif。
    verbose : bool
        是否打印加载信息。

    Returns
    -------
    list of dict : 每个文件的 PLIF 数据字典
    """
    results = []
    for f in file_list:
        try:
            data = load_plif(f, size, bit_shift, x0, y0, dx, dy)
            results.append(data)
            if verbose:
                print(f"  已加载 PLIF: {data['file_name']}")
        except Exception as e:
            if verbose:
                print(f"  加载失败 {f}: {e}")
    return results


def _apply_bin_average(arr: np.ndarray, bin_size: int) -> np.ndarray:
    bin_size = min(bin_size, *arr.shape)  # clip to array dimensions
    """2D 数组 bin 平均（下采样）"""
    if bin_size == 1:
        return arr
    ny, nx = arr.shape
    ny_new = max(1, ny // bin_size)
    nx_new = max(1, nx // bin_size)
    trimmed = arr[:ny_new * bin_size, :nx_new * bin_size]
    reshaped = trimmed.reshape(ny_new, bin_size, nx_new, bin_size)
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

    # 处理全 NaN 或全零输入
    if scalar_raw is None or scalar_raw.size == 0:
        if scalar_raw is not None:
            return np.zeros_like(scalar_raw, dtype='float32')
        return np.zeros((1, 1), dtype='float32')
    scalar_raw = np.nan_to_num(scalar_raw, nan=0.0)

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


def compute_plif_statistics(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算 PLIF 帧序列的统计信息。

    Returns
    -------
    dict : global_min, global_max, per_frame [{min, max, mean, std}, ...]
    """
    if not frames:
        return {}
    global_min = np.inf
    global_max = -np.inf
    per_frame = []
    for f in frames:
        s = f['scalar']
        stats = {
            'min': float(np.nanmin(s)),
            'max': float(np.nanmax(s)),
            'mean': float(np.nanmean(s)),
            'std': float(np.nanstd(s)),
        }
        global_min = min(global_min, stats['min'])
        global_max = max(global_max, stats['max'])
        per_frame.append(stats)
    return {
        'global_min': global_min, 'global_max': global_max,
        'per_frame': per_frame,
    }
