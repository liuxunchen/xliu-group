#!/usr/bin/env python3
"""
PLIF 原始 Raw 数据 (Photron FASTCAM TIFF) 读取与画图测试脚本。
"""

import sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.plif_loader import (
    parse_cihx, load_plif_raw_folder, load_plif,
    compute_plif_statistics, _read_tiff
)


TEST_DIR = Path(__file__).parent / "data" / "PLIF_raw"


def find_available_folder():
    """查找可用的 PLIF raw 文件夹"""
    if not TEST_DIR.exists():
        return None
    for d in sorted(TEST_DIR.glob("acetone_*")):
        if d.is_dir() and list(d.glob("*.cihx")):
            return str(d)
    return None


def test_parse_cihx():
    print("\n" + "=" * 60)
    print("[测试 1] parse_cihx - 解析 Photron 元数据")
    folder = find_available_folder()
    if not folder:
        print("  ! 未找到 PLIF 原始数据文件夹，跳过")
        return

    cihx_files = list(Path(folder).glob("*.cihx"))
    if not cihx_files:
        print("  ! 未找到 .cihx 文件，跳过")
        return

    meta = parse_cihx(cihx_files[0])
    assert meta['width'] > 0, f"width 应为正数: {meta['width']}"
    assert meta['height'] > 0, f"height 应为正数: {meta['height']}"
    assert meta['bit_depth'] > 0, f"bit_depth 应为正数: {meta['bit_depth']}"
    assert meta['record_rate'] > 0, f"record_rate 应为正数: {meta['record_rate']}"
    print(f"  ✓ 相机: {meta['camera_name']}")
    print(f"  ✓ 分辨率: {meta['width']} x {meta['height']}")
    print(f"  ✓ 位深: {meta['bit_depth']} bit, 有效: {meta['effective_depth']} bit ({meta['effective_side']})")
    print(f"  ✓ 帧率: {meta['record_rate']} fps, 快门: 1/{meta['shutter_speed']} s")
    print(f"  ✓ 总帧数: {meta['total_frame']}, 录制帧数: {meta['recorded_frame']}")


def test_load_plif_raw_folder():
    print("\n" + "=" * 60)
    print("[测试 2] load_plif_raw_folder - 加载原始文件夹")
    folder = find_available_folder()
    if not folder:
        print("  ! 未找到 PLIF 原始数据文件夹，跳过")
        return

    # 仅加载 3 帧用于快速测试
    result = load_plif_raw_folder(folder, max_frames=3, verbose=True)
    assert 'frames' in result, "应包含 'frames' key"
    assert 'metadata' in result, "应包含 'metadata' key"
    frames = result['frames']
    assert len(frames) == 3, f"应加载 3 帧，实际 {len(frames)}"

    # 检查每帧数据结构
    for i, f in enumerate(frames):
        assert 'scalar' in f, f"帧 {i} 缺少 'scalar'"
        assert 'X' in f, f"帧 {i} 缺少 'X'"
        assert 'Y' in f, f"帧 {i} 缺少 'Y'"
        assert f['scalar'].ndim == 2, f"帧 {i} scalar 应为 2D"
        assert f['scalar'].dtype == np.float32, f"帧 {i} 应为 float32"

    print(f"\n  ✓ 成功加载 {len(frames)} 帧")
    print(f"  ✓ 帧 0 形状: {frames[0]['scalar'].shape}")
    print(f"  ✓ 帧 0 范围: [{frames[0]['scalar'].min():.1f}, {frames[0]['scalar'].max():.1f}]")

    if result['metadata']:
        meta = result['metadata']
        print(f"  ✓ 元数据已解析: {meta['camera_name']}")


def test_load_single_tiff_raw():
    print("\n" + "=" * 60)
    print("[测试 3] load_plif (单张 TIFF, bit_shift)")
    folder = find_available_folder()
    if not folder:
        print("  ! 未找到数据，跳过")
        return

    tiff_files = sorted(Path(folder).glob("*.tif"))
    if not tiff_files:
        print("  ! 未找到 TIFF，跳过")
        return

    # 读取原始 uint16 数值对比
    raw = _read_tiff(str(tiff_files[0]))
    print(f"  原始 TIFF (uint16): shape={raw.shape}, min={raw.min()}, max={raw.max()}")

    # 不位移加载
    r0 = load_plif(str(tiff_files[0]), bit_shift=0)
    print(f"  不移位: shape={r0['scalar'].shape}, range=[{r0['scalar'].min():.1f}, {r0['scalar'].max():.1f}]")

    # 自动位移 (bit_shift=4 for 12-bit higher)
    r4 = load_plif(str(tiff_files[0]), bit_shift=4)
    print(f"  右移 4 位: range=[{r4['scalar'].min():.1f}, {r4['scalar'].max():.1f}]")

    # 验证移位正确: scalar /= 2^shift
    expected_max = raw.max() >> 4
    assert abs(r4['scalar'].max() - expected_max) < 1, \
        f"移位不一致: {r4['scalar'].max():.1f} != {expected_max}"
    print(f"  ✓ bit_shift=4 结果正确: max={r4['scalar'].max():.1f}")


def test_compute_statistics():
    print("\n" + "=" * 60)
    print("[测试 4] compute_plif_statistics")
    folder = find_available_folder()
    if not folder:
        print("  ! 未找到数据，跳过")
        return

    result = load_plif_raw_folder(folder, max_frames=5, verbose=False)
    stats = compute_plif_statistics(result['frames'])
    assert 'global_min' in stats, "应包含 global_min"
    assert 'global_max' in stats, "应包含 global_max"
    assert 'per_frame' in stats, "应包含 per_frame"
    assert len(stats['per_frame']) == 5, f"应有 5 帧统计，实际 {len(stats['per_frame'])}"

    print(f"  ✓ 全局范围: [{stats['global_min']:.1f}, {stats['global_max']:.1f}]")
    for i, s in enumerate(stats['per_frame']):
        print(f"  帧 {i}: min={s['min']:.1f}, max={s['max']:.1f}, "
              f"mean={s['mean']:.1f}, std={s['std']:.1f}")
    print("  ✓ 统计计算通过")


def test_bit_shift_parameter():
    print("\n" + "=" * 60)
    print("[测试 5] load_plif bit_shift 参数兼容性")
    # 使用模拟 TIFF 测试
    test_file = Path("/tmp/test_plif_bit_shift.tif")
    try:
        import cv2
        fake_img = np.random.randint(0, 65535, (50, 80), dtype=np.uint16)
        cv2.imwrite(str(test_file), fake_img)
    except Exception:
        print("  ! 无法创建模拟文件，跳过")
        return

    # 不移位
    r0 = load_plif(str(test_file), bit_shift=0)
    # 移 4 位
    r4 = load_plif(str(test_file), bit_shift=4)

    # 验证: r4 的标量约为 r0 的标量右移 4 位
    diff = np.abs(r0['scalar'] - (r4['scalar'].astype('float64') * 16))
    max_err = diff.max()
    print(f"  ✓ bit_shift=0 vs 4 最大误差: {max_err:.1f} (应在 ±15 以内)")
    assert max_err < 16, f"移位误差过大: {max_err}"
    print("  ✓ bit_shift 参数正常工作")

    test_file.unlink(missing_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("PLIF 原始 (Photron FASTCAM) 数据加载测试")
    print("=" * 60)

    test_parse_cihx()
    test_load_plif_raw_folder()
    test_load_single_tiff_raw()
    test_compute_statistics()
    test_bit_shift_parameter()

    print("\n" + "=" * 60)
    print("所有 PLIF 原始数据测试完成")
    print("=" * 60)
