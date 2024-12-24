import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from model.compared_results.add_std_deviation import dataset_names
from model.data_util import get_ecoli2

# 数据集名称列表
# 24个数据集
# dataset_names = ['ecoli2', 'satimage', 'optical_digits', 'pen_digits', 'spectrometer',
#                  'isolet', 'us_crime', 'libras_move', 'thyroid_sick', 'solar_flare_m0',
#                  'oil', 'wine_quality', 'letter_img', 'yeast_me2', 'ozone_level', 'WDBC',
#                  'yeast3', 'yeast4', 'yeast5', 'waveform1', 'waveform2', 'waveform3',
#                  'statlog_vehicle_silhouettes2', 'statlog_vehicle_silhouettes4']

dataset_names = ['ecoli2']

#
# 模型名称列表（指定顺序）
model_order = [
    "SelfPacedEnsembleClassifier",
    "BalanceCascadeClassifier",
    "UnderBaggingClassifier",
    "EasyEnsembleClassifier",
    "RUSBoostClassifier",
    "BalancedRandomForestClassifier",
    "AdaCostClassifier",
    "AdaUBoostClassifier",
    "AsymBoostClassifier",
    "CatBoostClassifier",
    "SMOTEBoostClassifier",
    "OverBaggingClassifier",
    "OverBoostClassifier",
    "SMOTEBaggingClassifier",
    "KmeansSMOTEBoostClassifier",
    "UncertaintyAwareDeepForest",
]

# 处理所有数据集和模型
def process_all_results(base_path):
    results = []  # 用于存储所有结果

    # 遍历每个数据集
    for dataset_idx, dataset_name in enumerate(dataset_names):
        dataset_path = os.path.join(base_path, f"{dataset_name}_result")
        print(f"Processing dataset: {dataset_name}")

        # 遍历每个模型
        for model_name in model_order:
            f1_scores, auc_scores, aupr_scores = [], [], []


            y_true = None
            y_pred = None
            y_proba = None


            # 遍历每个 fold
            for fold in range(1, 6):  # 5 folds
                fold_path = os.path.join(dataset_path, f"fold_{fold}")
                true_file = os.path.join(fold_path, f"{dataset_name}_{model_name}_true_label.npy")
                pred_file = os.path.join(fold_path, f"{dataset_name}_{model_name}_pred.npy")
                proba_file = os.path.join(fold_path, f"{dataset_name}_{model_name}_proba.npy")

                # 检查文件是否存在
                if not (os.path.exists(true_file) and os.path.exists(pred_file) and os.path.exists(proba_file)):
                    print(f"Missing files for {model_name} in fold {fold}")
                    continue

                # 加载数据并拼接
                true_labels = np.load(true_file)
                pred_labels = np.load(pred_file)
                proba_values = np.load(proba_file)

                y_true = true_labels if y_true is None else np.concatenate((y_true, true_labels))
                y_pred = pred_labels if y_pred is None else np.concatenate((y_pred, pred_labels))
                y_proba = proba_values if y_proba is None else np.concatenate((y_proba, proba_values))

            # 检查是否所有 folds 都加载成功
            if y_true is None or y_pred is None or y_proba is None:
                print("Data loading failed. Ensure all files exist.")
            else:
                print("Data loaded successfully.")
                print(f"y_true shape: {y_true.shape}")
                print(f"y_pred shape: {y_pred.shape}")
                print(f"y_proba shape: {y_proba.shape}")

        # 计算 b d u
        for i in range(len(y_true)):
            S = 4
            if y_true[i] == 1:
                b = (y_pred[i] + 1) / 4
                d = (2 - y_pred[i]) / 4
                u = 2 / S
            elif y_true[i] == 0:
                d = y_pred[i]
                b = 1 - y_pred[i]
            else:




# 设置根目录路径
if __name__ == "__main__":
    base_path = "./"  # 当前路径
    process_all_results(base_path)
    get_ecoli2()
