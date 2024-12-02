from sympy.physics.quantum.gate import CPHASE
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
    dataset = fetch_ucirepo(id=39)
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

def get_glass1():
    dataset = fetch_ucirepo(id=42)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 5 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("Glass1 类别分布:", Counter(y))
    return X, y, "Glass1"

def get_glass2():
    dataset = fetch_ucirepo(id=42)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("Glass2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 7 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("Glass2 类别分布:", Counter(y))
    return X, y, "Glass2"

def get_glass3():
    dataset = fetch_ucirepo(id=42)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("Glass3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 1 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("Glass3 类别分布:", Counter(y))
    return X, y, "Glass3"

def get_glass4():
    dataset = fetch_ucirepo(id=42)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("Glass4 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 3 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("Glass4 类别分布:", Counter(y))
    return X, y, "Glass4"


def get_glass5():
    dataset = fetch_ucirepo(id=42)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("Glass5 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 2 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("Glass5 类别分布:", Counter(y))
    return X, y, "Glass5"


if __name__ == "__main__":
    get_glass5()
