import numpy as np
from preprocess import data_load


def sample_entropy(U, m, r):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        B = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1.0) / (N - m) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(B)

    N = len(U)

    return -np.log(_phi(m + 1) / _phi(m))


def cal_se(eeg, interval):
    i = 0
    group = []
    while i < 7500:
        group.append(eeg[i:i + interval])
        i += interval
    se = [sample_entropy(seg, 2, 0.2 * np.std(seg, ddof=1)) for seg in group]

    return np.mean(se)


def se_analyse(data, interval):
    stage_se = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    for file in data:
        curr_stage = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            se = cal_se(eeg, interval)
            stage_se[stage].append(se)
            curr_stage[stage].append(se)

        print("---------------- " + file + " situation ----------------")
        for stage in curr_stage:
            mean = np.mean(curr_stage[stage])
            std = np.std(curr_stage[stage])
            print(stage, "mean", mean, ", std", std)
        print("\n")

    print("---------------- total situation ----------------")
    for stage in stage_se:
        mean = np.mean(stage_se[stage])
        std = np.std(stage_se[stage])
        print(stage, "mean", mean, ", std", std)


if __name__ == "__main__":
    data = data_load.load_data()
    se_analyse(data, 1500)
