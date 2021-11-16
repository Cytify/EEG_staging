import numpy as np
from preprocess.clean import load_data
from sample_entropy.se import sample_entropy


def multiscale_entropy(data, m, t):
    coarse = []
    for i in range(0, len(data), t):
        sum = np.sum(data[i:i + t])
        coarse.append(sum / t * 1.0)

    return sample_entropy(coarse, m, 0.2 * np.std(coarse, ddof=1))


def cal_mse(eeg, interval, t):
    i = 0
    group = []
    while i < 7500:
        group.append(eeg[i:i + interval])
        i += interval
    se = [multiscale_entropy(seg, 2, t) for seg in group]

    return np.mean(se)


def mse_analyse(data, interval, t):
    stage_mse = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    for file in data:
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            se = cal_mse(eeg, interval, t)
            stage_mse[stage].append(se)

    print(stage_mse)


if __name__ == "__main__":
    data = load_data()
    # t的选择有多种，可以分析，论文中提出，t最好在11与12之间
    mse_analyse(data, 750, 11)
