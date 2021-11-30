import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate, KFold
import pickle
from preprocess import data_load
from feature import amplitude, energy_ratio, pse, mse, se, hjorth, kc_complexity
import sklearn.svm as svm
import sklearn.ensemble as ensemble
from sklearn.neural_network import MLPClassifier


def get_feature():
    data = data_load.load_data()
    eeg_data = []
    eeg_target = []
    for file in data:
        print("current file:", file)
        i=0
        total=len(data[file])
        for seg in data[file]:
            eeg = seg[0]
            label = seg[1]

            # amp = amplitude.amplitude(eeg)
            spectral_entropy = pse.spectral_entropy(eeg, 250)
            delta_ratio = energy_ratio.energy_ratio(eeg, 250, 1, 4)
            theta_ratio = energy_ratio.energy_ratio(eeg, 250, 4, 8)
            alpha_ratio = energy_ratio.energy_ratio(eeg, 250, 8, 13)
            beta_ratio = energy_ratio.energy_ratio(eeg, 250, 13, 30)
            sample_entropy = se.cal_se(eeg, 1500)
            multi_sample_entropy = mse.cal_mse(eeg, 7500, 11)
            [hjorth_mobility, hjorth_complexity] = hjorth.hjorth(eeg)
            kc = kc_complexity.kc_complexity(eeg)

            eeg_data.append([spectral_entropy, delta_ratio, beta_ratio, hjorth_mobility, theta_ratio, alpha_ratio,
                             sample_entropy, multi_sample_entropy, kc])

            eeg_target.append(label)
            i+=1
            if i%30==0:
                print(str(round(i/total*100))+"%")


    return eeg_data, eeg_target


def decision_tree(data, target):
    # iris = load_iris()
    # print(iris)
    # print(iris.data)
    # print(iris.target)
    # model = tree.DecisionTreeClassifier(criterion="gini")
    # model.fit(iris.data, iris.target)
    # save_tree("tree.pickle", model)

    # 交叉验证划分训练集和测试集.test_size为测试集所占的比例
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)
    print('训练集大小：', x_train.shape, y_train.shape)  # 训练集样本大小
    print('测试集大小：', x_test.shape, y_test.shape)  # 测试集样本大小

    print("model train start")

    # model = tree.DecisionTreeClassifier(criterion="gini")
    # model = svm.SVC(kernel='linear')
    # model = svm.SVC(kernel='poly', degree=3)
    model = svm.SVC(kernel='rbf', C=600)
    # model = ensemble.RandomForestClassifier()
    # model = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
    model = model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print("model train end")
    print('准确率：', score)  # 计算测试集的度量值（准确率）

    with open("score.txt", "a+") as f:
        f.write(str(score) + "\n")

    print("save model")
    save_tree("tree.pickle", model)

    # kf = KFold(n_splits=20)
    # i = 1
    # for train_index, test_index in kf.split(data):
    #     print(str(i) + " th ford:")
    #     train_start = train_index[0]
    #     train_end = train_index[-1]
    #     test_start = test_index[0]
    #     test_end = test_index[-1]
    #
    #     x_train = data[train_start:test_start] + data[test_end:train_end]
    #     # y_train =
    #     x_test = data[test_start:test_end]
    #     # y_test =
    #     print('train_index %s, test_index %s'%(train_index, test_index))
    #     break


def save_tree(path, model):
    # 保存模型
    with open(path, "wb") as f:
        pickle.dump(model, f)


def predict_tree(path, data):
    # 读取模型
    with open(path, "rb") as f:
        model = pickle.load(f)
    predicted = model.predict(data)  # 对新数据进行预测
    print('predictedY:' + str(predicted))  # 输出为predictedY:[1]，表示愿意购买，1表示yes


def visualize_tree(model):
    pass


if __name__ == "__main__":
    eeg_data, eeg_target = get_feature()
    eeg_data = np.asarray(eeg_data)
    eeg_target = np.asarray(eeg_target)
    print(eeg_data)
    print(eeg_target)

    decision_tree(eeg_data, eeg_target)
