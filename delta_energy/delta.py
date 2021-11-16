import numpy as np
from scipy.fftpack import fft
from preprocess.clean import load_data


def delta_energy_ratio(sig, fs):
    # fs 采样率
    fr = fft(sig)
    n = len(fr)

    fr = np.abs(fr)
    x = np.arange(0, n) * fs / n
    hz_unit = int(n / fs)
    delta = np.trapz(fr[hz_unit:4 * hz_unit + 1], x[hz_unit:4 * hz_unit + 1])
    all = np.trapz(fr[hz_unit:30 * hz_unit + 1], x[hz_unit:30 * hz_unit + 1])

    return delta / all


def delta_analyse(data, fs):
    stage_delta = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    for file in data:
        print("current file:", file)
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            se = delta_energy_ratio(eeg, fs)
            stage_delta[stage].append(se)

    print(stage_delta)

    for stage in stage_delta:
        mean = np.mean(stage_delta[stage])
        std = np.std(stage_delta[stage])
        print(stage, "mean:", mean, "std:", std)


if __name__ == "__main__":
    data = load_data()
    delta_analyse(data, 250)
