import numpy as np
from preprocess import data_load


def kc_complexity(eeg):
    # 先二值化
    m = np.mean(eeg)
    sequence = ""
    for i in eeg:
        sequence += "1" if i > m else "0"

    sub_strings = set()
    n = len(sequence)

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = sequence[ind : ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1

    cn = len(sub_strings)
    bn = n / np.log2(n)

    return cn / bn


def kc_analyse(data):
    stage_kc = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    for file in data:
        curr_stage = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            kc = kc_complexity(eeg)
            stage_kc[stage].append(kc)
            curr_stage[stage].append(kc)

        print("---------------- " + file + " situation ----------------")
        for stage in curr_stage:
            mean = np.mean(curr_stage[stage])
            std = np.std(curr_stage[stage])
            print(stage, "mean", mean, ", std", std)
        print("\n")

    print("---------------- total situation ----------------")
    for stage in stage_kc:
        mean = np.mean(stage_kc[stage])
        std = np.std(stage_kc[stage])
        print(stage, "mean", mean, ", std", std)


if __name__ == "__main__":
    data = data_load.load_data()
    kc_analyse(data)
