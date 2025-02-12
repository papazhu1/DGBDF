from imbens.datasets import fetch_datasets
from sklearn.model_selection import StratifiedKFold
from UADF import UncertaintyAwareDeepForest
from demo import get_config
import pandas as pd
import numpy as np
from imbens.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from model.data_util import get_yeast5, get_glass3

if __name__ == "__main__":
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    # dataset_name = "solar_flare_m0"
    # dataset = fetch_datasets()[dataset_name]
    # X, y = dataset['data'], dataset['target']
    # y = np.where(y == -1, 0, y)
    X, y, name = get_glass3()
    f1_macro_list = []
    auc_list = []
    aupr_list = []
    gmean_list = []


    for train_idx, test_idx in skf.split(X, y):
        X_train_cv, X_val_cv = X[train_idx], X[test_idx]
        y_train_cv, y_val_cv = y[train_idx], y[test_idx]
        for lamb in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
            model = UncertaintyAwareDeepForest(get_config(lamb))
            model.fit(X_train_cv,y_train_cv)

            # 预测
            y_pred = model.predict(X_val_cv)
            y_pred_proba = model.predict_proba(X_val_cv)[:, 1]  # 获取预测的概率值，用于计算 AUC 和 AUPR

            # 计算性能指标
            f1_macro = f1_score(y_val_cv, y_pred, average='macro')
            auc = roc_auc_score(y_val_cv, y_pred_proba)
            aupr = average_precision_score(y_val_cv, y_pred_proba)
            gmean = geometric_mean_score(y_val_cv, y_pred)

            # 保存每轮的性能指标
            f1_macro_list.append(f1_macro)
            auc_list.append(auc)
            aupr_list.append(aupr)
            gmean_list.append(gmean)

            # 输出当前训练集比例和各项指标
            print(f"lambda: {lamb}")
            print(f"F1-macro: {f1_macro:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"AUPR: {aupr:.4f}")
            print(f"Gmean: {gmean:.4f}")
            print("-" * 40)

    n = 11
    f1_macro_list = np.array(f1_macro_list).reshape(-1, n)
    f1_macro_list = np.mean(f1_macro_list, axis=0)
    auc_list = np.array(auc_list).reshape(-1, n)
    auc_list = np.mean(auc_list, axis=0)
    aupr_list = np.array(aupr_list).reshape(-1, n)
    aupr_list = np.mean(aupr_list, axis=0)
    gmean_list = np.array(gmean_list).reshape(-1, n)
    gmean_list = np.mean(gmean_list, axis=0)

    print(f1_macro_list)
    print(auc_list)
    print(aupr_list)
    print(gmean_list)



