import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# 数据集名称列表
# 24个数据集
dataset_names = ['ecoli2', 'satimage', 'optical_digits', 'pen_digits', 'spectrometer',
                 'isolet', 'us_crime', 'libras_move', 'thyroid_sick', 'solar_flare_m0',
                 'oil', 'wine_quality', 'letter_img', 'yeast_me2', 'ozone_level', 'WDBC',
                 'yeast3', 'yeast4', 'yeast5', 'waveform1', 'waveform2', 'waveform3',
                 'statlog_vehicle_silhouettes2', 'statlog_vehicle_silhouettes4']

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

# 计算性能指标
def calculate_metrics(y_true, y_pred, y_proba):
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_proba)
    aupr = average_precision_score(y_true, y_proba)
    return f1, auc, aupr

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

                # 加载数据
                y_true = np.load(true_file)
                y_pred = np.load(pred_file)
                y_proba = np.load(proba_file)

                # 确保 proba 是正类的概率
                if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
                    y_proba = y_proba[:, 1]

                # 计算指标
                f1, auc, aupr = calculate_metrics(y_true, y_pred, y_proba)
                f1_scores.append(f1)
                auc_scores.append(auc)
                aupr_scores.append(aupr)

            # 计算均值和标准差
            if f1_scores:  # 确保有结果
                f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
                auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
                aupr_mean, aupr_std = np.mean(aupr_scores), np.std(aupr_scores)

                # 保存结果
                results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "F1-macro": f"{f1_mean:.4f} ± {f1_std:.4f}",
                    "AUC": f"{auc_mean:.4f} ± {auc_std:.4f}",
                    "AUPR": f"{aupr_mean:.4f} ± {aupr_std:.4f}"
                })

    # 转换为 DataFrame
    results_df = pd.DataFrame(results)

    # 按指定顺序排序模型
    results_df['Model'] = pd.Categorical(results_df['Model'], categories=model_order, ordered=True)
    results_df = results_df.sort_values(by=["Dataset", "Model"])

    # 保存到 Excel 文件
    output_file = "result_add_std_deviation.xlsx"
    results_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

# 设置根目录路径
if __name__ == "__main__":
    base_path = "./"  # 当前路径
    process_all_results(base_path)
