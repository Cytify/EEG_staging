import numpy as np
from preprocess import data_load
from scipy.fftpack import fft, ifft


def spectral_entropy(data, fs):
    fr = fft(data)
    n = len(fr)
    # 信号功率谱密度
    psd = [abs(i) ** 2 / n for i in fr]
    f = np.arange(0, len(fr)) * fs / n
    # 用0.3-35hz归一化功率谱密度
    f1 = int(0.3 * n / fs)
    f2 = int(35 * n / fs)
    base = np.trapz(psd[f1:f2 + 1], np.arange(f1, f2 + 1))
    pnsd = [i / base for i in psd]
    # 计算pse
    pse = np.trapz(pnsd[f1:f2 + 1] * np.log(pnsd[f1:f2 + 1])) / (-np.log(f2 - f1 + 1))

    return pse


def cal_pse(eeg, interval):
    i = 0
    group = []
    while i < 7500:
        group.append(eeg[i:i + interval])
        i += interval
    pse = [spectral_entropy(seg, 250) for seg in group]

    return np.mean(pse)


def pse_analyse(data, interval):
    stage_pse = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    for file in data:
        curr_stage = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            pse = cal_pse(eeg, interval)
            stage_pse[stage].append(pse)
            curr_stage[stage].append(pse)

        print("---------------- " + file + " situation ----------------")
        for stage in curr_stage:
            mean = np.mean(curr_stage[stage])
            std = np.std(curr_stage[stage])
            print(stage, "mean", mean, ", std", std)
        print("\n")

    print("---------------- total situation ----------------")
    for stage in stage_pse:
        mean = np.mean(stage_pse[stage])
        std = np.std(stage_pse[stage])
        print(stage, "mean", mean, ", std", std)


if __name__ == "__main__":
    data = data_load.load_data()
    pse_analyse(data, 7500)
