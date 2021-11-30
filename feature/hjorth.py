import numpy as np
from preprocess import data_load


def hjorth(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided,
           recommended to speed up)
    In case 1, D is computed using Numpy's Difference function.
    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    Parameters
    ----------
    X
        list
        a time series
    D
        list
        first order differential sequence of a time series
    Returns
    -------
    As indicated in return line
    Hjorth mobility and complexity
    """

    if D is None:
        D = np.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = np.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return np.sqrt(M2 / TP), np.sqrt(float(M4) * TP / M2 / M2)  # Hjorth Mobility and Complexity


def hjorth_analyse(data):
    stage_m = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    stage_x = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
    for file in data:
        curr_stage_m = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
        curr_stage_x = {"W": [], "1": [], "2": [], "3": [], "4": [], "R": []}
        for sig in data[file]:
            eeg = sig[0]
            stage = sig[1]
            [m, x] = hjorth(eeg)
            stage_m[stage].append(m)
            stage_x[stage].append(x)
            curr_stage_m[stage].append(m)
            curr_stage_x[stage].append(x)

        print("---------------- " + file + " situation ----------------")
        print("mobility")
        for stage in curr_stage_m:
            mean = np.mean(curr_stage_m[stage])
            std = np.std(curr_stage_m[stage])
            print(stage, "mean", mean, ", std", std)
        print("\n")

        print("---------------- " + file + " situation ----------------")
        print("complexity")
        for stage in curr_stage_x:
            mean = np.mean(curr_stage_x[stage])
            std = np.std(curr_stage_x[stage])
            print(stage, "mean", mean, ", std", std)
        print("\n")

    print("---------------- total situation ----------------")
    print("mobility")
    for stage in stage_m:
        mean = np.mean(stage_m[stage])
        std = np.std(stage_m[stage])
        print(stage, "mean", mean, ", std", std)

    print("---------------- total situation ----------------")
    print("complexity")
    for stage in stage_x:
        mean = np.mean(stage_x[stage])
        std = np.std(stage_x[stage])
        print(stage, "mean", mean, ", std", std)


if __name__ == "__main__":
    data = data_load.load_data()
    # t的选择有多种，可以分析，论文中提出，t最好在11与12之间
    hjorth_analyse(data)