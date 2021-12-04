import wfdb
import pyedflib
from preprocess import filter
import numpy as np
import csv


def split_eeg(eeg, ann, sample):
    stage_list = ["W", "1", "2", "3", "4", "R"]
    data = []
    # sample[0] = 0
    for index, value in enumerate(sample):
        stage = ann[index].split(" ")[0].strip(b'\x00'.decode())
        if stage not in stage_list:
            continue
        data.append([eeg[value:value + 3000], stage])

    return data


def read_edf(eeg_file, ann_file):
    eeg = pyedflib.EdfReader(eeg_file).readSignal(0)
    ann = []
    sample = []
    with open(ann_file) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            sample += [x for x in range(int(row[2]) * 100, (int(row[2]) + int(row[3])) * 100, 3000)]
            num = int(row[3]) / 30
            ann += row[4].split(" ")[-1] * int(num)

    return eeg, ann, sample


def load_data():
    path = r"D:/WorkSpace/PyCharmProject/EEG_staging/data/"
    # path = r"I:/ChromeDownload/mit-bih-polysomnographic-database-1.0.0/"
    # files = ["slp01a", "slp01b", "slp32", "slp41", "slp45", "slp59"]
    files = []
    with open(path + "RECORDS", "r") as f:
        for line in f.readlines():
            files.append(line[:-1])

    # 数组存放格式
    # data[filename][record],记录中[0]为原始信号，[1]为对应睡眠标记
    data = {}
    for file in files:
        # eeg = wfdb.rdrecord(path + file, channels=[2]).p_signal.reshape(-1)
        # 滤波
        # filter_eeg = filter.butter_low_trap_filter(eeg, 0.5, 35, 250)
        # z-score 标准化
        # filter_eeg = (filter_eeg - np.mean(filter_eeg)) / np.std(filter_eeg)
        # ann = wfdb.rdann("I:/ChromeDownload/mit-bih-polysomnographic-database-1.0.0/" + file, "st")
        # data[file] = split_eeg(filter_eeg, ann.aux_note, ann.sample)

        eeg, ann, sample = read_edf(path + file + ".edf", path + file + "_ann.txt")
        filter_eeg = filter.butter_low_trap_filter(eeg, 0.5, 35, 100)
        filter_eeg = (filter_eeg - np.mean(filter_eeg)) / np.std(filter_eeg)
        data[file] = split_eeg(filter_eeg, ann, sample)

    return data

