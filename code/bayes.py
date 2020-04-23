import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter


class NaiveBayes:
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(X):
        """计算均值
        Param: X : list or np.ndarray

        Return:
            avg : float

        """
        avg = 0.0
        # ========= show me your code ==================
        avg = sum(X) / float(len(X))
        # ========= show me your code ==================
        return avg

    # 标准差（方差）
    def stdev(self, X):
        """计算标准差
        Param: X : list or np.ndarray

        Return:
            res : float

        """
        res = 0.0
        # ========= show me your code ==================
        avg = self.mean(X)
        res = math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))
        # ========= show me your code ==================
        return res

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        """根据均值和标注差计算x符号该高斯分布的概率
        Parameters:
        ----------
        x : 输入
        mean : 均值
        stdev : 标准差

        Return:

        res : float， x符合的概率值

        """
        res = 0.0
        # ========= show me your code ==================
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        res = (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
        # ========= show me your code ==================

        return res

    # 处理X_train
    def summarize(self, train_data):
        """计算每个类目下对应数据的均值和标准差
        Param: train_data : list

        Return : [mean, stdev]
        """
        summaries = [0.0, 0.0]
        # ========= show me your code ==================
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]

        # ========= show me your code ==================
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {
            label: self.summarize(value) for label, value in data.items()
        }
        return 'gaussianNB train done!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        """计算数据在各个高斯分布下的概率
        Paramter:
        input_data : 输入数据

        Return:
        probabilities : {label : p}
        """
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        # ========= show me your code ==================
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], mean, stdev)
        # ========= show me your code ==================
        return probabilities

    # 类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(self.calculate_probabilities(
            X_test).items(), key=lambda x: x[-1])[-1][0]
        return label
    # 计算得分

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width',
                  'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:, :-1], data[:, -1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = NaiveBayes()
model.fit(X_train, y_train)
print(f'predict([4.4,  3.2,  1.3,  0.2]):{model.predict([4.4,  3.2,  1.3,  0.2])}')
print(f'xtest score:{model.score(X_test, y_test)}')

# 调用模型
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)

clf.score(X_test, y_test)
clf.predict([[4.4,  3.2,  1.3,  0.2]])
