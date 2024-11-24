from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml, load_iris, load_diabetes
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter

def get_heart_disease():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print(Counter(y))
    return X, y, "heart_disease"

def get_iris():
    data = load_iris()
    X, y = data.data, data.target
    return X, y, "iris"

# wisconsin diagnostic breast cancer 数据集
def get_WDBC():
    heart_disease = fetch_ucirepo(id=17)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print(Counter(y))
    return X, y, "WDBC"

# Wine1 数据集
def get_wine1():
    dataset = fetch_ucirepo(id=109)  # 假设 Wine 数据集的 ID 为 9
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print("Wine 类别分布:", Counter(y))
    # 将标签转换为二分类：2, 3 为少数类，其余为多数类
    y = y.apply(lambda x: 0 if x in [2, 3] else 1)  # 1 表示少数类, 0 表示多数类
    print(y)

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    print("Wine 类别分布:", Counter(y))
    return X, y, "wine1"

# Wine1 数据集
def get_wine2():
    dataset = fetch_ucirepo(id=109)  # 假设 Wine 数据集的 ID 为 9
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print("Wine 类别分布:", Counter(y))
    # 将标签转换为二分类：2, 3 为少数类，其余为多数类
    y = y.apply(lambda x: 1 if x in [1, 2] else 0)  # 1 表示少数类, 0 表示多数类
    print(y)

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    print("Wine 类别分布:", Counter(y))
    return X, y, "wine2"

# ionosphere 数据集
def get_ionosphere():
    data = fetch_openml("ionosphere", version=1)
    X, y = data.data, data.target
    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y, "ionosphere"

# Ecoli 数据集
def get_ecoli():
    dataset = fetch_ucirepo(id=39)  # 假设 Ecoli 数据集的 ID 为 10
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    print("Ecoli 类别分布:", Counter(y))
    return X, y, "ecoli"


# Ecoli1 数据集
def get_ecoli1():
    dataset = fetch_ucirepo(id=39)  # 假设 Ecoli 数据集的 ID 为 10
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print("Ecoli 类别分布:", Counter(y))

    # 将标签转换为二分类：im 为少数类，其余为多数类
    y = y.apply(lambda x: 1 if x == 'im' or x == 'pp' else 0)  # 1 表示少数类, 0 表示多数类

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    print("Ecoli1 类别分布:", Counter(y))
    return X, y, "ecoli1"

# Ecoli2 数据集
def get_ecoli2():
    dataset = fetch_ucirepo(id=39)  # 假设 Ecoli 数据集的 ID 为 10
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 将标签转换为二分类：im 为少数类，其余为多数类
    y = y.apply(lambda x: 1 if x == 'imU' or x == 'pp' or x == 'om' else 0)  # 1 表示少数类, 0 表示多数类

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    print("Ecoli2 类别分布:", Counter(y))
    return X, y, "ecoli2"


if __name__ == "__main__":
    get_ecoli1()
