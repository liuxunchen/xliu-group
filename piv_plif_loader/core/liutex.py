import numpy as np

def liutex_2d(u, v, dx=1.0, dy=1.0, signed=True):
    """
    计算二维不可压缩流场的 Liutex 强度及相关物理量。

    Parameters
    ----------
    u, v : 2D ndarray (ny, nx)
        速度分量。
    dx, dy : float
        网格间距。
    signed : bool
        是否返回带符号的 Liutex（旋转方向由涡量符号决定）。

    Returns
    -------
    R : ndarray (ny, nx)
        Liutex 强度（若 signed=True 则带符号，否则非负）。
    S : ndarray
        剪切贡献。
    lambda_ci : ndarray
        局部旋转角速度（复数特征值的虚部）。
    omega : ndarray
        涡量 (v_x - u_y)。
    """
    # 梯度计算
    uy, ux = np.gradient(u, dy, dx, axis=(0, 1))
    vy, vx = np.gradient(v, dy, dx, axis=(0, 1))

    omega = vx - uy
    lambda_ci_sq = np.maximum(ux * vy - uy * vx, 0.0)
    lambda_ci = np.sqrt(lambda_ci_sq)

    discriminant = np.maximum(omega**2 - 4.0 * lambda_ci_sq, 0.0)
    sqrt_disc = np.sqrt(discriminant)

    R_abs = np.abs(omega) - sqrt_disc

    if signed:
        R = np.sign(omega) * R_abs
    else:
        R = R_abs

    S = sqrt_disc
    return R, S, lambda_ci, omega