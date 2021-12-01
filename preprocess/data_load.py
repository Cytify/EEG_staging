import wfdb
from preprocess import filter
import numpy as np


def split_eeg(eeg, ann, sample):
    stage_list = ["W", "1", "2", "3", "4", "R"]
    data = []
    sample[0] = 0
    for index, value in enumerate(sample):
        stage = ann[index].split(" ")[0].strip(b'\x00'.decode())
        if stage not in stage_list:
            continue
        data.append([eeg[value:value + 7500], stage])

    return data


def load_data():
    files = ["slp01a", "slp01b", "slp32", "slp41", "slp45", "slp59"]
    # files = []
    # with open("D:/WorkSpace/PyCharmProject/EEG_staging/psg_data/RECORDS", "r") as f:
    #     for line in f.readlines():
    #         files.append(line[:-1])

    # 数组存放格式
    # data[filename][record],记录中[0]为原始信号，[1]为对应睡眠标记
    data = {}
    for file in files:
        eeg = wfdb.rdrecord("D:/WorkSpace/PyCharmProject/EEG_staging/psg_data/" + file, channels=[2]).p_signal.reshape(
            -1)
        # 滤波
        filter_eeg = filter.butter_low_trap_filter(eeg, 0.5, 35, 250)
        # z-score 标准化
        # filter_eeg = filter_eeg - np.mean(filter_eeg)
        # filter_eeg = filter_eeg / np.std(filter_eeg)

        ann = wfdb.rdann("D:/WorkSpace/PyCharmProject/EEG_staging/psg_data/" + file, "st")
        data[file] = split_eeg(filter_eeg, ann.aux_note, ann.sample)

    return data
