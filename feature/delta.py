import numpy as np
from scipy.fftpack import fft
from preprocess import data_load


def delta_energy_ratio(sig, fs):
    # fs 采样率
    fr = fft(sig)
    n = len(fr)

    fr = [abs(i) ** 2 for i in fr]
    x = np.arange(0, n) * fs / n
    hz_unit = int(n / fs)
    delta = np.trapz(fr[hz_unit:4 * hz_unit + 1], x[hz_unit:4 * hz_unit + 1])
    all = np.trapz(fr[hz_unit:30 * hz_unit + 1], x[hz_unit:30 * hz_unit + 1])

    return delta / all


def delta_analyse(data, fs):
    stage_delta = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    for file in data:
        curr_stage = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            delta = delta_energy_ratio(eeg, fs)
            stage_delta[stage].append(delta)
            curr_stage[stage].append(delta)

        print("---------------- " + file + " situation ----------------")
        for stage in curr_stage:
            mean = np.mean(curr_stage[stage])
            std = np.std(curr_stage[stage])
            print(stage, "mean", mean, ", std", std)
        print("\n")

    print("---------------- total situation ----------------")
    for stage in stage_delta:
        mean = np.mean(stage_delta[stage])
        std = np.std(stage_delta[stage])
        print(stage, "mean", mean, ", std", std)


if __name__ == "__main__":
    data = data_load.load_data()
    delta_analyse(data, 250)
