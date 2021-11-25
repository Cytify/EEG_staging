import numpy as np
from preprocess import data_load


# 求幅值
def amplitude(sig):
    sig.sort()

    return np.mean(sig[-10:]) - np.mean(sig[:10])


def amp_analyse(data):
    stage_amp = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    for file in data:
        curr_stage = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            se = amplitude(eeg)
            stage_amp[stage].append(se)
            curr_stage[stage].append(se)

        print("---------------- " + file + " situation ----------------")
        for stage in curr_stage:
            mean = np.mean(curr_stage[stage])
            std = np.std(curr_stage[stage])
            print(stage, "mean", mean, ", std", std)
        print("\n")

    print("---------------- total situation ----------------")
    for stage in stage_amp:
        mean = np.mean(stage_amp[stage])
        std = np.std(stage_amp[stage])
        print(stage, "mean", mean, ", std", std)


if __name__ == "__main__":
    data = data_load.load_data()
    amp_analyse(data)
