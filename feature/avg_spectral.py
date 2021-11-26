import numpy as np
from scipy.fftpack import fft
from preprocess import data_load


def avg_spectral(sig, fs):
    fr = fft(sig)
    n = len(fr)
    # 信号功率谱密度
    psd = [abs(i) ** 2 / n for i in fr]
    f = np.arange(0, len(fr)) * fs / n

    avg_sp = np.trapz(psd, f)

    return avg_sp


def avg_spectral_analyse(data, fs):
    stage_avg_pectral = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    for file in data:
        curr_stage = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            avg_sp = avg_spectral(eeg, fs)
            stage_avg_pectral[stage].append(avg_sp)
            curr_stage[stage].append(avg_sp)

        print("---------------- " + file + " situation ----------------")
        for stage in curr_stage:
            mean = np.mean(curr_stage[stage])
            std = np.std(curr_stage[stage])
            print(stage, "mean", mean, ", std", std)
        print("\n")

    print("---------------- total situation ----------------")
    for stage in stage_avg_pectral:
        mean = np.mean(stage_avg_pectral[stage])
        std = np.std(stage_avg_pectral[stage])
        print(stage, "mean", mean, ", std", std)


if __name__ == "__main__":
    data = data_load.load_data()
    avg_spectral_analyse(data, 250)
