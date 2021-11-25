from scipy import signal
from sklearn.decomposition import FastICA


def median_filter(sig, wdn):
    median_res = signal.medfilt(sig, wdn)
    return median_res


def butter_low_trap_filter(sig, low_fr, high_fr, fs):
    b, a = signal.butter(5, 2 * high_fr / fs, "lowpass")
    butter_filter = signal.filtfilt(b, a, sig)
    b, a = signal.butter(5, 2 * low_fr / fs, "highpass")
    butter_filter = signal.filtfilt(b, a, butter_filter)

    return butter_filter


def ica_filter(sig):
    ica = FastICA(n_components=4)
