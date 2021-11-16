import numpy as np
from preprocess.clean import load_data
import entropy as ent


def spectral_entropy(data, fs):
    return ent.spectral_entropy(data, fs, method="fft", normalize=True)


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
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            se = cal_pse(eeg, interval)
            stage_pse[stage].append(se)

    print(stage_pse)


if __name__ == "__main__":
    data = load_data()
    pse_analyse(data, 750)
