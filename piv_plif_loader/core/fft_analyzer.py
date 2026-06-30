import numpy as np

def point_fft(signal, dt=1.0):
    """
    对输入信号进行 FFT 并返回半谱频率和幅值。

    Parameters
    ----------
    signal : 1D array
        时间序列。
    dt : float
        采样时间间隔。

    Returns
    -------
    freqs : ndarray
        频率数组（正频率部分）。
    amplitude : ndarray
        对应的幅值（单边谱）。
    """
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=dt)
    half = n // 2
    # 跳过 0 Hz (DC 分量)
    amplitude = np.abs(fft_vals[1:half]) * 2.0 / n
    return freqs[1:half], amplitude
