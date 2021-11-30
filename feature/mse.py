import numpy as np
from preprocess import data_load
from feature import se


def multiscale_entropy(data, m, t):
    coarse = []
    for i in range(0, len(data), t):
        sum = np.sum(data[i:i + t])
        coarse.append(sum / t * 1.0)

    return se.samp_entropy(coarse, m, 0.2 * np.std(coarse, ddof=1))


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
        curr_stage = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            mse = cal_mse(eeg, interval, t)
            stage_mse[stage].append(mse)
            curr_stage[stage].append(mse)

        print("---------------- " + file + " situation ----------------")
        for stage in curr_stage:
            mean = np.mean(curr_stage[stage])
            std = np.std(curr_stage[stage])
            print(stage, "mean", mean, ", std", std)
        print("\n")

    print("---------------- total situation ----------------")
    for stage in stage_mse:
        mean = np.mean(stage_mse[stage])
        std = np.std(stage_mse[stage])
        print(stage, "mean", mean, ", std", std)


if __name__ == "__main__":
    data = data_load.load_data()
    # t的选择有多种，可以分析，论文中提出，t最好在11与12之间
    mse_analyse(data, 7500, 11)
