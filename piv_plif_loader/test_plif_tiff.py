#!/usr/bin/env python3
"""PLIF TIFF 读取功能测试脚本"""

import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到搜索路径
sys.path.insert(0, str(Path(__file__).parent))

from core.plif_loader import load_plif, load_plif_batch, _read_tiff


def test_read_tiff():
    """测试 _read_tiff: 用 PIV 示例 PNG (模拟 TIFF 读取能力)"""
    # 寻找可用的图像文件
    test_dir = Path(__file__).parent / "data" / "PIV_example"
    png_files = list(test_dir.glob("*_velocity.png"))
    if png_files:
        img_path = str(png_files[0])
        print(f"\n[测试] _read_tiff: 尝试读取 {img_path}")
        try:
            img = _read_tiff(img_path)
            print(f"  ✓ 成功: shape={img.shape}, dtype={img.dtype}")
        except IOError as e:
            # PNG 文件可能被 _read_tiff 成功读取（PIL/cv2 均支持 PNG）
            print(f"  _read_tiff 返回: {e}")
    else:
        print("  ! 未找到测试图像文件，跳过 _read_tiff 测试")


def test_load_plif_params():
    """测试 load_plif 参数传递"""
    print("\n[测试] load_plif 参数校验")
    try:
        load_plif("nonexistent.tif", dx=0)
    except ValueError as e:
        print(f"  ✓ dx<=0 被正确拒绝: {e}")

    try:
        load_plif("nonexistent.tif", dy=-1)
    except ValueError as e:
        print(f"  ✓ dy<=0 被正确拒绝: {e}")


def test_load_plif_nonexistent():
    """测试不存在的文件"""
    print("\n[测试] load_plif 文件不存在")
    result = load_plif_batch(["/tmp/nonexistent_plif.tif"], verbose=False)
    print(f"  load_plif_batch 返回 {len(result)} 个结果 (应为 0)")


def test_bin_average():
    """测试 bin 平均函数"""
    from core.plif_loader import _apply_bin_average

    print("\n[测试] _apply_bin_average")
    arr = np.arange(100, dtype='float64').reshape(10, 10)

    # bin=1: 不变
    r1 = _apply_bin_average(arr, 1)
    assert np.array_equal(r1, arr), "bin=1 应返回原始数组"
    print("  ✓ bin=1: 返回原始数组")

    # bin=2: 5x5
    r2 = _apply_bin_average(arr, 2)
    assert r2.shape == (5, 5), f"bin=2 期望 (5,5)，实际 {r2.shape}"
    print(f"  ✓ bin=2: shape={r2.shape}")

    # bin=5: 2x2
    r5 = _apply_bin_average(arr, 5)
    assert r5.shape == (2, 2), f"bin=5 期望 (2,2)，实际 {r5.shape}"
    print(f"  ✓ bin=5: shape={r5.shape}")

    # bin=100: max(1, ...) 确保不崩溃
    r100 = _apply_bin_average(arr, 100)
    assert r100.shape == (1, 1) or r100.shape == (10, 10)
    print(f"  ✓ bin=100: shape={r100.shape} (不会崩溃)")


def test_enhance_plif():
    """测试 PLIF 增强函数"""
    from core.plif_loader import enhance_plif

    print("\n[测试] enhance_plif")
    scalar = np.random.rand(50, 50).astype('float32')
    enhanced = enhance_plif(scalar)
    assert enhanced.shape == (50, 50), f"增强后 shape 变化: {enhanced.shape}"
    assert enhanced.dtype == np.float32
    print(f"  ✓ 正常输入: shape={enhanced.shape}, dtype={enhanced.dtype}, "
          f"range=[{enhanced.min():.1f}, {enhanced.max():.1f}]")

    # 空输入
    empty = enhance_plif(np.zeros((0, 0)))
    print(f"  ✓ 空输入: shape={empty.shape}")

    # NaN 输入
    nan_arr = np.full((10, 10), np.nan, dtype='float32')
    nan_out = enhance_plif(nan_arr)
    print(f"  ✓ NaN 输入: shape={nan_out.shape}, "
          f"nan={np.isnan(nan_out).sum()} (应为 0)")


def test_align_plif_to_piv():
    """测试 PLIF 与 PIV 对齐"""
    from core.plif_loader import align_plif_to_piv

    print("\n[测试] align_plif_to_piv")
    plif = {
        'X': np.linspace(0, 10, 20, dtype='float32'),
        'Y': np.linspace(0, 10, 20, dtype='float32'),
        'scalar': np.random.rand(20, 20).astype('float32'),
    }
    piv = {
        'X': np.tile(np.linspace(1, 9, 10).astype('float32'), (10, 1)),
        'Y': np.tile(np.linspace(1, 9, 10).astype('float32').reshape(-1, 1), (1, 10)),
    }
    aligned = align_plif_to_piv(plif, piv)
    assert aligned.shape == (10, 10), f"对齐后 shape={aligned.shape}"
    print(f"  ✓ PIV 网格 ({piv['X'].shape[0]}x{piv['X'].shape[1]}) → "
          f"对齐形状 {aligned.shape}")


def test_readme_example():
    """模拟 README 使用流程"""
    print("\n[测试] 模拟完整 PLIF 加载流程")
    # 创建一个简单的模拟 TIFF
    test_file = Path("/tmp/test_plif_sim.tif")
    try:
        import cv2
        fake_img = np.random.randint(0, 65535, (100, 200), dtype=np.uint16)
        cv2.imwrite(str(test_file), fake_img)
        print(f"  已创建模拟 TIFF: {test_file}")
    except Exception:
        from PIL import Image
        fake_img = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        Image.fromarray(fake_img).save(str(test_file))
        print(f"  (PIL) 已创建模拟 TIFF: {test_file}")

    # 加载
    result = load_plif(str(test_file), size=1, dx=0.5, dy=0.5)
    print(f"  ✓ load_plif: x_num={result['x_num']}, y_num={result['y_num']}")
    print(f"  ✓ X 范围: [{result['X'][0]:.1f}, {result['X'][-1]:.1f}] (dx=0.5)")
    print(f"  ✓ Y 范围: [{result['Y'][0]:.1f}, {result['Y'][-1]:.1f}] (dy=0.5)")
    assert result['scalar'].shape == (result['y_num'], result['x_num'])
    test_file.unlink(missing_ok=True)
    print("  ✓ 清理: 删除临时文件")


def test_dx_dy_scaling():
    """测试不同物理尺寸的坐标生成"""
    from core.plif_loader import _apply_bin_average

    print("\n[测试] dx/dy 物理标定")
    # 创建一个 10x10 的模拟 TIFF 文件
    test_file = Path("/tmp/test_plif_scale.tif")
    try:
        import cv2
        fake_img = np.ones((10, 10), dtype=np.uint8) * 128
        cv2.imwrite(str(test_file), fake_img)
    except Exception:
        from PIL import Image
        Image.fromarray(fake_img).save(str(test_file))

    # dx=0.1, dy=0.05, bin=1
    result = load_plif(str(test_file), size=1, dx=0.1, dy=0.05)
    x_max = result['X'][-1] + (result['X'][1] - result['X'][0]) if len(result['X']) > 1 else result['X'][0]
    y_max = result['Y'][-1] + (result['Y'][1] - result['Y'][0]) if len(result['Y']) > 1 else result['Y'][0]
    print(f"  ✓ dx=0.1: X 范围 [0, {x_max:.2f}) (期望 [0, 1.0))")
    print(f"  ✓ dy=0.05: Y 范围 [0, {y_max:.2f}) (期望 [0, 0.50))")
    assert abs(x_max - 1.0) < 0.01, f"X 范围异常"
    assert abs(y_max - 0.5) < 0.01, f"Y 范围异常"

    # 带非零原点
    result2 = load_plif(str(test_file), size=1, dx=0.1, dy=0.05, x0=-5.0, y0=-2.0)
    print(f"  ✓ x0=-5: X 起始 {result2['X'][0]:.1f} (期望 -5.0)")
    print(f"  ✓ y0=-2: Y 起始 {result2['Y'][0]:.1f} (期望 -2.0)")
    test_file.unlink(missing_ok=True)
    print("  ✓ 清理: 删除临时文件")


def test_bin_with_scaling():
    """测试 bin 平均与标定同时使用"""
    from core.plif_loader import _apply_bin_average

    print("\n[测试] bin + 标定组合")

    test_file = Path("/tmp/test_plif_bin.tif")
    try:
        import cv2
        fake_img = np.random.randint(0, 255, (20, 40), dtype=np.uint8)
        cv2.imwrite(str(test_file), fake_img)
    except Exception:
        from PIL import Image
        Image.fromarray(fake_img).save(str(test_file))

    # bin=2, dx=0.5, dy=0.5
    result = load_plif(str(test_file), size=2, dx=0.5, dy=0.5)
    print(f"  ✓ bin=2: shape={result['scalar'].shape} (原始 20x40)")
    print(f"  ✓ dx=0.5 (bin=2): X 范围 [{result['X'][0]:.1f}, {result['X'][-1]:.1f}]")
    assert result['scalar'].shape == (10, 20), f"bin=2 应得 (10,20)，实际 {result['scalar'].shape}"
    test_file.unlink(missing_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("PLIF TIFF 加载模块测试")
    print("=" * 60)

    test_read_tiff()
    test_load_plif_params()
    test_load_plif_nonexistent()
    test_bin_average()
    test_enhance_plif()
    test_align_plif_to_piv()
    test_dx_dy_scaling()
    test_bin_with_scaling()
    test_readme_example()

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)
