import numpy as np
from scipy.fftpack import fft
from preprocess import data_load


def energy_ratio(sig, fs, start_fr, end_fr):
    # fs 采样率
    fr = fft(sig)
    n = len(fr)

    fr = [abs(i) ** 2 for i in fr]
    x = np.arange(0, n) * fs / n
    hz_unit = int(n / fs)
    delta = np.trapz(fr[start_fr * hz_unit:end_fr * hz_unit + 1], x[start_fr * hz_unit:end_fr * hz_unit + 1])
    all = np.trapz(fr[hz_unit:30 * hz_unit + 1], x[hz_unit:30 * hz_unit + 1])

    return delta / all


def delta_analyse(data, fs, start_fr, end_fr):
    stage_delta = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    for file in data:
        curr_stage = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            delta = energy_ratio(eeg, fs, start_fr, end_fr)
            stage_delta[stage].append(delta)
            curr_stage[stage].append(delta)

        # print("---------------- " + file + " situation ----------------")
        # for stage in curr_stage:
        #     mean = np.mean(curr_stage[stage])
        #     std = np.std(curr_stage[stage])
        #     print(stage, "mean", mean, ", std", std)
        # print("\n")

    print("---------------- total situation ----------------")
    for stage in stage_delta:
        mean = np.mean(stage_delta[stage])
        std = np.std(stage_delta[stage])
        print(stage, "mean", mean, ", std", std)


if __name__ == "__main__":
    data = data_load.load_data()
    delta_analyse(data, 250, 1, 4)  # delta
    delta_analyse(data, 250, 4, 8)  # theta
    delta_analyse(data, 250, 8, 13)  # alpha
    delta_analyse(data, 250, 13, 30)  # beta
