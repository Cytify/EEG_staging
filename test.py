# # import numpy as np
# # import scipy.ndimage
# # import scipy.signal
# #
# # X = np.array([1, 2, 3]).reshape(1,3)  # input data
# # H = np.array([4, 5, 6]).reshape(1,3)  # kernel
# #
# # # display scipy.signal.correlate2 results
# # print ("scipy.signal.correlate2d results")
# # print ("full: {}".format(scipy.signal.correlate2d(X, H, 'full')))
# # print ("same: {}".format(scipy.signal.correlate2d(X, H, 'same')))
# # print ("valid: {}".format(scipy.signal.correlate2d(X, H, 'valid')))
# #
# # print ('')
# # # display scipy.signal.convolve2 results
# # print ("scipy.signal.convolve2d results")
# # print ("full: {}".format(scipy.signal.convolve2d(X, H, 'full')))
# # print ("same: {}".format(scipy.signal.convolve2d(X, H, 'same')))
# # print ("valid: {}".format(scipy.signal.convolve2d(X, H, 'valid')))
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import FastICA
#
# C = 200  # 样本数
# x = np.arange(C)
# s1 = 2 * np.sin(0.02 * np.pi * x)  # 正弦信号
#
# a = np.linspace(-2, 2, 25)
# s2 = np.array([a, a, a, a, a, a, a, a]).reshape(200, )  # 锯齿信号
# s3 = np.array(20 * (5 * [2] + 5 * [-2]))  # 方波信号
# s4 = 4 * (np.random.random([1, C]) - 0.5).reshape(200, )  # 随机信号
# """画出4种信号"""
# ax1 = plt.subplot(411)
# ax2 = plt.subplot(412)
# ax3 = plt.subplot(413)
# ax4 = plt.subplot(414)
# ax1.plot(x, s1)
# ax2.plot(x, s2)
# ax3.plot(x, s3)
# ax4.plot(x, s4)
# plt.show()
# """将4种信号混合
# 其中mix矩阵是混合后的信号矩阵，shape=[4,200]"""
# s = np.array([s1, s2, s3, s4])
# ran = 2 * np.random.random([4, 4])
# mix = ran.dot(s)
# ax1 = plt.subplot(411)
# ax2 = plt.subplot(412)
# ax3 = plt.subplot(413)
# ax4 = plt.subplot(414)
# ax1.plot(x, mix[0, :])
# ax2.plot(x, mix[1, :])
# ax3.plot(x, mix[2, :])
# ax4.plot(x, mix[3, :])
# plt.show()
#
# ica = FastICA(n_components=4)  # 独立成分为4个
# mix = mix.T  # 将信号矩阵转为[n_samples,n_features],即[200,4]
# u = ica.fit_transform(mix)  # u为解混后的4个独立成分，shape=[200,4]
# u = u.T  # shape=[4,200]
# print(ica.n_iter_)  # 算法迭代次数
# """画出解混后的4种独立成分"""
# ax1 = plt.subplot(411)
# ax2 = plt.subplot(412)
# ax3 = plt.subplot(413)
# ax4 = plt.subplot(414)
# ax1.plot(x, u[0, :])
# ax2.plot(x, u[1, :])
# ax3.plot(x, u[2, :])
# ax4.plot(x, u[3, :])
# plt.show()


from sklearn.model_selection import train_test_split, cross_val_score, cross_validate  # 交叉验证所需的函数
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, ShuffleSplit  # 交叉验证所需的子集划分方法
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit  # 分层分割
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, LeavePGroupsOut, GroupShuffleSplit  # 分组分割
from sklearn.model_selection import TimeSeriesSplit  # 时间序列分割
from sklearn import datasets  # 自带数据集
from sklearn import svm  # SVM算法
from sklearn import preprocessing  # 预处理模块
from sklearn.metrics import recall_score  # 模型度量

iris = datasets.load_iris()  # 加载数据集
print('样本集大小：', iris.data.shape, iris.target.shape)

# ===================================数据集划分,训练模型==========================
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4,
                                                    random_state=0)  # 交叉验证划分训练集和测试集.test_size为测试集所占的比例
print('训练集大小：', X_train.shape, y_train.shape)  # 训练集样本大小
print('测试集大小：', X_test.shape, y_test.shape)  # 测试集样本大小
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)  # 使用训练集训练模型
print('准确率：', clf.score(X_test, y_test))  # 计算测试集的度量值（准确率）

#  如果涉及到归一化，则在测试集上也要使用训练集模型提取的归一化函数。
scaler = preprocessing.StandardScaler().fit(X_train)  # 通过训练集获得归一化函数模型。（也就是先减几，再除以几的函数）。在训练集和测试集上都使用这个归一化函数
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(kernel='linear', C=1).fit(X_train_transformed, y_train)  # 使用训练集训练模型
X_test_transformed = scaler.transform(X_test)
print(clf.score(X_test_transformed, y_test))  # 计算测试集的度量值（准确度）

# ===================================直接调用交叉验证评估模型==========================
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)  # cv为迭代次数。
print(scores)  # 打印输出每次迭代的度量值（准确度）
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）

# ===================================多种度量结果======================================
scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring, cv=5, return_train_score=True)
sorted(scores.keys())
print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）

# ==================================K折交叉验证、留一交叉验证、留p交叉验证、随机排列交叉验证==========================================
# k折划分子集
kf = KFold(n_splits=2)
for train, test in kf.split(iris.data):
    print("k折划分：%s %s" % (train.shape, test.shape))
    break

# 留一划分子集
loo = LeaveOneOut()
for train, test in loo.split(iris.data):
    print("留一划分：%s %s" % (train.shape, test.shape))
    break

# 留p划分子集
lpo = LeavePOut(p=2)
for train, test in loo.split(iris.data):
    print("留p划分：%s %s" % (train.shape, test.shape))
    break

# 随机排列划分子集
ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
for train_index, test_index in ss.split(iris.data):
    print("随机排列划分：%s %s" % (train.shape, test.shape))
    break

# ==================================分层K折交叉验证、分层随机交叉验证==========================================
skf = StratifiedKFold(n_splits=3)  # 各个类别的比例大致和完整数据集中相同
for train, test in skf.split(iris.data, iris.target):
    print("分层K折划分：%s %s" % (train.shape, test.shape))
    break

skf = StratifiedShuffleSplit(n_splits=3)  # 划分中每个类的比例和完整数据集中的相同
for train, test in skf.split(iris.data, iris.target):
    print("分层随机划分：%s %s" % (train.shape, test.shape))
    break

# ==================================组 k-fold交叉验证、留一组交叉验证、留 P 组交叉验证、Group Shuffle Split==========================================
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

# k折分组
gkf = GroupKFold(n_splits=3)  # 训练集和测试集属于不同的组
for train, test in gkf.split(X, y, groups=groups):
    print("组 k-fold分割：%s %s" % (train, test))

# 留一分组
logo = LeaveOneGroupOut()
for train, test in logo.split(X, y, groups=groups):
    print("留一组分割：%s %s" % (train, test))

# 留p分组
lpgo = LeavePGroupsOut(n_groups=2)
for train, test in lpgo.split(X, y, groups=groups):
    print("留 P 组分割：%s %s" % (train, test))

# 随机分组
gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
for train, test in gss.split(X, y, groups=groups):
    print("随机分割：%s %s" % (train, test))

# ==================================时间序列分割==========================================
tscv = TimeSeriesSplit(n_splits=3)
TimeSeriesSplit(max_train_size=None, n_splits=3)
for train, test in tscv.split(iris.data):
    print("时间序列分割：%s %s" % (train, test))