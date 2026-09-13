import numpy as np

def compute_gradients(u, v, dx, dy):
    """计算速度梯度分量 ux, uy, vx, vy"""
    uy, ux = np.gradient(u, dy, dx, axis=(0, 1))
    vy, vx = np.gradient(v, dy, dx, axis=(0, 1))
    return ux, uy, vx, vy

def compute_all_fields(u, v, dx, dy):
    """
    返回一个包含常用物理量的字典：
    - velocity_mag: 合速度
    - vorticity: 涡量 (curl z)
    - divergence: 散度
    - grad_u: |∇u|
    - grad_v: |∇v|
    """
    ux, uy, vx, vy = compute_gradients(u, v, dx, dy)
    velocity_mag = np.sqrt(u**2 + v**2)
    vorticity = vx - uy
    divergence = ux + vy
    grad_u = np.sqrt(ux**2 + uy**2)
    grad_v = np.sqrt(vx**2 + vy**2)


    return {
        'Velocity magnitude': velocity_mag,
        'Vorticity': vorticity,
        'Divergence': divergence,
        'Grad U': grad_u,
        'Grad V': grad_v
    }

def compute_scalar_gradient(scalar, dx, dy):
    """计算标量场的梯度模"""
    sy, sx = np.gradient(scalar, dy, dx, axis=(0, 1))
    return np.sqrt(sx**2 + sy**2)
