import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft


def median_filter(sig, wdn):
    median_res = signal.medfilt(sig, wdn)

    # print(len(sig))
    # wdn = int(wdn / 2) * 2 + 1
    # sig_len = len(sig)
    # median_res = []
    # half = int(wdn / 2)
    # for i in range(0, half):
    #     wd = sig[:i + 1]
    #     wd.sort()
    #     median_res.append(np.median(wd))
    #
    # for i in range(half, sig_len - half):
    #     if i % 100000 == 0:
    #         print(i)
    #     wd = sig[i - half:i + half + 1]
    #     wd.sort()
    #     median_res.append(np.median(wd))
    #
    # for i in range(sig_len - half, sig_len):
    #     wd = sig[sig_len - half:i + 1]
    #     wd.sort()
    #     median_res.append(np.median(wd))
    #
    return median_res


def butter_low_trap_filter(sig, cutoff_fr):
    b, a = signal.butter(5, 0.6, "lowpass")
    butter_filter = signal.filtfilt(b, a, sig)
    b, a = signal.iirnotch(50 / 125, 30)
    pass


# fs = 1000
# t = np.arange(fs) / fs
# s_50hz = np.sin(2*np.pi*50*t)
# s_100hz = np.cos(2*np.pi*100*t)
# s = s_50hz + s_100hz
#
# b, a = signal.iirnotch(50, Q=30, fs=1000)
# y = signal.lfilter(b, a, s)
#
# plt.subplot(3,1,1)
# plt.ylabel('50Hz+100Hz')
# plt.plot(t, s)
#
# plt.subplot(3,1,2)
# plt.ylabel('100Hz')
# plt.plot(t, s_100hz)
#
# plt.subplot(3,1,3)
# plt.ylim(-1.1,1.1)
# plt.ylabel('filtered waveform')
# plt.plot(t, y)
# plt.show()
#
# plt.ylabel('filtered waveform')
# plt.plot(t, abs(fft(y)))
# plt.show()