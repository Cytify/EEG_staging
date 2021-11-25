import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import wfdb
import filter


def load_slp45():
    eeg = wfdb.rdrecord("D:/WorkSpace/PyCharmProject/EEG_staging/psg_data/slp01b", channels=[2]).p_signal.reshape(-1)
    return eeg


def show_data(sig):
    fs = 250
    n = len(sig)
    t = np.arange(0, n) / 250
    # 原始信号波形图
    plt.title("original signal")
    plt.plot(t, sig)
    plt.show()

    # 原始信号频率振幅图
    amp = fft(sig)
    fr = np.arange(0, n) * fs / n
    # 取一半
    amp_half = np.abs(amp)[:int(len(fr) / 2)]
    fr_half = fr[:int(len(fr) / 2)]
    plt.title("frequency amplitude")
    plt.plot(fr_half, amp_half)
    plt.show()

    # 信号滤波
    filter_eeg = filter.butter_low_trap_filter(sig, 0.5, 35, 250)
    plt.title("wavelet denoise signal")
    plt.plot(t, filter_eeg)
    plt.show()

    # 小波去噪信号频率振幅图
    amp = fft(filter_eeg)
    amp_half = np.abs(amp)[:int(len(fr) / 2)]
    plt.title("wavelet denoise frequency amplitude")
    plt.plot(fr_half, amp_half, linewidth=1.0)
    plt.show()


show_data(load_slp45())


