import numpy as np
from preprocess.clean import load_data


# 求幅值
def amplitude(sig):
    sig.sort()

    return np.mean(sig[-20:]) - np.mean(sig[:20])


def amp_analyse(data):
    stage_amp = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    for file in data:
        print("current file:", file)
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            se = amplitude(eeg)
            stage_amp[stage].append(se)

    print(stage_amp)

    for stage in stage_amp:
        mean = np.mean(stage_amp[stage])
        std = np.std(stage_amp[stage])
        print(stage, "mean:", mean, "std:", std)


if __name__ == "__main__":
    data = load_data()
    amp_analyse(data)
