from sympy.physics.quantum.gate import CPHASE
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml, load_iris, load_diabetes
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# 非01类型
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


# wisconsin diagnostic breast cancer 数据集
# 01类型
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
    print("ecoli.shape", X.shape)
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

# 01类型
def get_haberman():
    heart_disease = fetch_ucirepo(id=43)
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
    return X, y, "haberman"

def get_car1():
    dataset = fetch_ucirepo(id=19)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("car1 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == "acc" else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("car1 处理后类别分布:", Counter(y))
    return X, y, "car1"

def get_car2():
    dataset = fetch_ucirepo(id=19)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("car2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == "good" else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("car2 处理后类别分布:", Counter(y))
    return X, y, "car2"


def get_car3():
    dataset = fetch_ucirepo(id=19)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("car3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == "vgood" else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("car3 处理后类别分布:", Counter(y))
    return X, y, "car3"

# 01类型,但是数据太少了，用不了
def get_hepatitis():
    dataset = fetch_ucirepo(id=46)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("hepatitis 处理前类别分布:", Counter(y))

    # X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    # y = np.array([1 if label == "vgood" else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("hepatitis 处理后类别分布:", Counter(y))
    return X, y, "hepatitis"

def get_poker_hand():
    dataset = fetch_ucirepo(id=158)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("poker_hand 处理前类别分布:", Counter(y))

    # X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    # y = np.array([1 if label == "vgood" else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("poker_hand 处理后类别分布:", Counter(y))
    return X, y, "poker_hand"

def get_liver_disorders1():
    dataset = fetch_ucirepo(id=60)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("liver_disorders1 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 0.5 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("liver_disorders1 处理后类别分布:", Counter(y))
    return X, y, "liver_disorders1"

def get_liver_disorders2():
    dataset = fetch_ucirepo(id=60)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("liver_disorders2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 4.0 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("liver_disorders2 处理后类别分布:", Counter(y))
    return X, y, "liver_disorders2"

def get_liver_disorders3():
    dataset = fetch_ucirepo(id=60)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("liver_disorders3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 6.0 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("liver_disorders3 处理后类别分布:", Counter(y))
    return X, y, "liver_disorders3"

def get_liver_disorders4():
    dataset = fetch_ucirepo(id=60)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("liver_disorders4 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 2.0 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("liver_disorders4 处理后类别分布:", Counter(y))
    return X, y, "liver_disorders4"


def get_yeast1():
    dataset = fetch_ucirepo(id=110)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("yeast1 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'CYT' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("yeast1 处理后类别分布:", Counter(y))
    return X, y, "yeast1"

def get_yeast2():
    dataset = fetch_ucirepo(id=110)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("yeast2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'NUC' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("yeast2 处理后类别分布:", Counter(y))
    return X, y, "yeast2"

def get_yeast3():
    dataset = fetch_ucirepo(id=110)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("yeast3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'MIT' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("yeast3 处理后类别分布:", Counter(y))
    return X, y, "yeast3"

def get_yeast4():
    dataset = fetch_ucirepo(id=110)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("yeast4 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'ME3' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("yeast4 处理后类别分布:", Counter(y))
    return X, y, "yeast4"

def get_yeast5():
    dataset = fetch_ucirepo(id=110)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("yeast5 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'ME2' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("yeast5 处理后类别分布:", Counter(y))
    return X, y, "yeast5"

def get_waveform1():
    dataset = fetch_ucirepo(id=107)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("waveform1 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 0 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("waveform1 处理后类别分布:", Counter(y))
    return X, y, "waveform1"

def get_waveform2():
    dataset = fetch_ucirepo(id=107)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("waveform2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 1 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("waveform2 处理后类别分布:", Counter(y))
    return X, y, "waveform2"

def get_waveform3():
    dataset = fetch_ucirepo(id=107)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("waveform3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 2 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("waveform3 处理后类别分布:", Counter(y))
    return X, y, "waveform3"

def get_page_blocks1():
    dataset = fetch_ucirepo(id=78)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("page_blocks1 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 2 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("page_blocks1 处理后类别分布:", Counter(y))
    return X, y, "page_blocks1"

def get_page_blocks2():
    dataset = fetch_ucirepo(id=78)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("page_blocks2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 5 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("page_blocks2 处理后类别分布:", Counter(y))
    return X, y, "page_blocks2"


def get_page_blocks3():
    dataset = fetch_ucirepo(id=78)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("page_blocks3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 4 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("page_blocks3 处理后类别分布:", Counter(y))
    return X, y, "page_blocks3"


def get_page_blocks4():
    dataset = fetch_ucirepo(id=78)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("page_blocks4 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 3 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("page_blocks4 处理后类别分布:", Counter(y))
    return X, y, "page_blocks4"

def get_statlog_vehicle_silhouettes1():
    dataset = fetch_ucirepo(id=149)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("statlog_vehicle_silhouettes1 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'saab' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("statlog_vehicle_silhouettes1 处理后类别分布:", Counter(y))
    return X, y, "statlog_vehicle_silhouettes1"

def get_statlog_vehicle_silhouettes2():
    dataset = fetch_ucirepo(id=149)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("statlog_vehicle_silhouettes2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'bus' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("statlog_vehicle_silhouettes2 处理后类别分布:", Counter(y))
    return X, y, "statlog_vehicle_silhouettes2"

def get_statlog_vehicle_silhouettes3():
    dataset = fetch_ucirepo(id=149)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("statlog_vehicle_silhouettes3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'opel' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("statlog_vehicle_silhouettes3 处理后类别分布:", Counter(y))
    return X, y, "statlog_vehicle_silhouettes3"

def get_statlog_vehicle_silhouettes4():
    dataset = fetch_ucirepo(id=149)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("statlog_vehicle_silhouettes4 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'van' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("statlog_vehicle_silhouettes4 处理后类别分布:", Counter(y))
    return X, y, "statlog_vehicle_silhouettes4"


if __name__ == "__main__":
    get_statlog_vehicle_silhouettes1()
