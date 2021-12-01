import numpy as np
from preprocess import data_load


def amp_diff(eeg, threshold):
    diff_eeg = [abs(eeg[i] - eeg[i - 1]) for i in range(1, len(eeg))]
    return sum(i >= threshold for i in diff_eeg)


def amp_diff_analyse(data, threshold):
    stage_amp_diff = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    for file in data:
        curr_stage = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            diff = amp_diff(eeg, threshold)
            stage_amp_diff[stage].append(diff)
            curr_stage[stage].append(diff)

        print("---------------- " + file + " situation ----------------")
        for stage in curr_stage:
            mean = np.mean(curr_stage[stage])
            std = np.std(curr_stage[stage])
            print(stage, "mean", mean, ", std", std)
        print("\n")

    print("---------------- total situation ----------------")
    for stage in stage_amp_diff:
        mean = np.mean(stage_amp_diff[stage])
        std = np.std(stage_amp_diff[stage])
        print(stage, "mean", mean, ", std", std)


if __name__ == "__main__":
    data = data_load.load_data()
    amp_diff_analyse(data, 0.02)
