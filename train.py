from sklearn import tree
from sklearn.datasets import load_iris
import pickle


def decision_tree():
    iris = load_iris()
    print(iris)
    print(iris.data)
    print(iris.target)
    model = tree.DecisionTreeClassifier(criterion="gini")
    model.fit(iris.data, iris.target)
    save_tree("tree.pickle", model)


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