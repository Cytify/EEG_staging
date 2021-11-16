import math
import pywt
import numpy as np
import wfdb


def wextend(x, len_ext):
    len_x = len(x)
    ext_x = []
    for i in range(len_ext - 1, -1, -1):
        ext_x.append(x[i])
    ext_x = ext_x + x
    for i in range(len_x - 1, len_x - len_ext - 1, -1):
        ext_x.append(x[i])

    return ext_x


def wconv1(x, f, shape):
    return np.convolve(x, f, shape)


def dwt(low_d, high_d, x):
    lf = len(low_d)
    lx = len(x)
    first = 1
    len_ext = lf - 1
    last = lx + lf - 1

    y = wextend(x, len_ext)

    z = wconv1(y, low_d, "valid")
    a = [z[i] for i in range(first, last, 2)]

    z = wconv1(y, high_d, "valid")
    d = [z[i] for i in range(first, last, 2)]

    return [a, d]
    pass


def wavedec(sig, n, wavelet):
    low_d = [-0.0105974017849973, 0.0328830116669829, 0.0308413818359870, -0.187034811718881, -0.0279837694169839, 0.630880767929590, 0.714846570552542, 0.230377813308855]
    high_d = [-0.230377813308855, 0.714846570552542, -0.630880767929590, -0.0279837694169839, 0.187034811718881, 0.0308413818359870, -0.0328830116669829, -0.0105974017849973]

    s = [len(sig), 1]
    l = np.zeros(n + 2)
    l[-1] = s[0]
    c = []

    x = sig.tolist()
    for k in range(0, n):
        [x, d] = dwt(low_d, high_d, x)
        l[n - k] = len(d)
        c = d + c

    c = x + c
    l[0] = len(x)

    return [c, l]


def waverec(c, l, wavelet):
    coeffs = []
    index = 0
    for i in l[:-1]:
        print(i)
        coeffs.append(np.array(c[index:index + i]))
        index += i

    return pywt.waverec(coeffs, wavelet)


def pywavedec(sig, n, wavelet):
    wave = pywt.wavedec(sig, "db1", level=n)
    c = []
    l = []
    for w in wave:
        c += w.tolist()
        l.append(len(w))
    l.append(len(sig))

    return [c, l]


def ddencmp(sig):
    n = len(sig)
    thr = math.sqrt(2 * math.log(n))
    [c, l] = pywavedec(sig, 1, "db1")
    c = c[l[0]:]
    normaliz = np.median(np.abs(c))
    thr = thr * normaliz / 0.6745

    return thr


def wdencmp(sig, n, wavelet, thr):
    [c, l] = wavedec(sig, n, wavelet)
    inddet = np.arange(int(l[0]), len(c))
    res = wthresh(c[int(l[0]):], thr)
    for index, i in enumerate(inddet):
        c[i] = res[index]
    print(c[:50])
    return waverec(c, l, wavelet)


def wthresh(sig, thr):
    tmp = [d - thr for d in np.abs(sig)]
    tmp = [(d + abs(d)) / 2 for d in tmp]
    tmp = [i * j for i, j in zip(tmp, sig)]

    return tmp


def wavelet_denoise(sig, n, wavelet):
    thr = ddencmp(sig)
    print(thr)
    # wdencmp 函数debug
    denoise_sig = wdencmp(sig, n, wavelet, thr)

    return denoise_sig


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

    # 数组存放格式
    # data[filename][record],记录中[0]为原始信号，[1]为对应睡眠标记
    data = {}
    for file in files:
        eeg = wfdb.rdrecord("D:/WorkSpace/PyCharmProject/EEG_staging/psg_data/" + file, channels=[2]).p_signal.reshape(
            -1)
        # 去除高频
        eeg = remove_high_fr(eeg, 8, "db4")
        # 小波阈值去噪
        eeg = wavelet_denoise(eeg, 8, "db4")

        ann = wfdb.rdann("D:/WorkSpace/PyCharmProject/EEG_staging/psg_data/" + file, "st")
        data[file] = split_eeg(eeg, ann.aux_note, ann.sample)

    return data


def remove_high_fr(sig, n, wavelet):
    [ca8, cd8, cd7, cd6, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(sig, wavelet, level=n)
    ca8 = np.zeros(len(ca8))
    cd1 = np.zeros(len(cd1))
    #cd2 = np.zeros(len(cd2))
    coeffs = [ca8, cd8, cd7, cd6, cd5, cd4, cd3, cd2, cd1]
    filted_data = pywt.waverec(coeffs, wavelet)

    return filted_data



def waveletdec(self, s, coef_type='d', wname='sym7', level=6, mode='symmetric'):
    import pywt
    N = len(s)
    w = pywt.Wavelet(wname)
    a = s
    ca = []
    cd = []
    for i in range(level):
        (a, d) = pywt.dwt(a, w, mode)  # 将a作为输入进行dwt分解
        ca.append(a)
        cd.append(d)
    rec_a = []
    rec_d = []
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w)[0:N])  # 进行重构
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w)[0:N])  # 进行重构
    if coef_type == 'd':
        return rec_d
    return rec_a
