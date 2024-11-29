import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
from imbens.metrics import geometric_mean_score, sensitivity_score, specificity_score
from imbens.ensemble import (
    SelfPacedEnsembleClassifier, BalanceCascadeClassifier, UnderBaggingClassifier,
    EasyEnsembleClassifier, RUSBoostClassifier, BalancedRandomForestClassifier,
    AdaCostClassifier, AdaUBoostClassifier, AsymBoostClassifier
)
from sklearn.model_selection import StratifiedKFold
from imbens.datasets import fetch_datasets
from collections import Counter
import os

# 加载数据集
def load_data(dataset_name):
    dataset = fetch_datasets()[dataset_name]
    X, y = dataset['data'], dataset['target']
    y = np.where(y == -1, 0, y)  # 将 -1 类别转换为 0
    print(f"Original class distribution: {Counter(dataset['target'])}")
    print(f"Transformed class distribution: {Counter(y)}")
    return X, y

# 评估模型性能并存储预测结果
def evaluate_and_save_predictions(model, X, y, dataset_name, model_name, n_splits=5):
    # 创建以数据集名称为文件夹名称的目录
    save_dir = f"{dataset_name}_result"
    os.makedirs(save_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, precs, recs, gmeans, aucs, auprs, sens, spes = [], [], [], [], [], [], [], [], []

    # 存储所有样本的预测结果
    all_proba = []
    all_pred = []
    all_true_label = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 保存当前折的预测结果
        all_proba.extend(y_pred_proba)
        all_pred.extend(y_pred)
        all_true_label.extend(y_test)

        # 计算性能指标
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='macro'))
        precs.append(precision_score(y_test, y_pred, average='macro'))
        recs.append(recall_score(y_test, y_pred, average='macro'))
        gmeans.append(geometric_mean_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_pred_proba))
        auprs.append(average_precision_score(y_test, y_pred_proba))
        sens.append(sensitivity_score(y_test, y_pred))
        spes.append(specificity_score(y_test, y_pred))

    # 保存预测结果到 .npy 文件
    np.save(os.path.join(save_dir, f"{dataset_name}_{model_name}_proba.npy"), np.array(all_proba))
    np.save(os.path.join(save_dir, f"{dataset_name}_{model_name}_pred.npy"), np.array(all_pred))
    np.save(os.path.join(save_dir, f"{dataset_name}_{model_name}_true_label.npy"), np.array(all_true_label))
    print(f"Saved predictions for {model_name} to {save_dir}")

    results = {
        "Accuracy": np.mean(accs),
        "F1": np.mean(f1s),
        "Precision": np.mean(precs),
        "Recall": np.mean(recs),
        "G-mean": np.mean(gmeans),
        "AUC": np.mean(aucs),
        "AUPR": np.mean(auprs),
        "Sensitivity": np.mean(sens),
        "Specificity": np.mean(spes),
    }
    return results


def save_and_print_results_table(results, dataset_name):
    # 确保列顺序
    column_order = ["Model", "acc", "sen", "spe", "f1_macro", "gmean", "auc", "aupr", "precision", "recall"]
    save_dir = f"{dataset_name}_result"
    os.makedirs(save_dir, exist_ok=True)

    # 检查是否存在之前的结果文件
    csv_filename = os.path.join(save_dir, f"{dataset_name}_results.csv")
    if os.path.exists(csv_filename):
        # 如果存在，加载旧结果
        existing_results = pd.read_csv(csv_filename)
        print(f"Loaded existing results from {csv_filename}")
    else:
        # 如果不存在，创建一个空的 DataFrame
        existing_results = pd.DataFrame(columns=column_order)

    # 将新的结果转换为 DataFrame
    new_results_df = pd.DataFrame(results).round(4)

    # 合并新旧结果，以模型名称为基准更新旧结果
    combined_results = pd.concat([existing_results, new_results_df]).drop_duplicates(subset="Model", keep="last")

    # 排序结果，保持与模型列表顺序一致
    model_order = [
        "SelfPacedEnsembleClassifier",
        "BalanceCascadeClassifier",
        "UnderBaggingClassifier",
        "EasyEnsembleClassifier",
        "RUSBoostClassifier",
        "BalancedRandomForestClassifier",
        "AdaCostClassifier",
        "AdaUBoostClassifier",
        "AsymBoostClassifier"
    ]
    combined_results["Model"] = pd.Categorical(combined_results["Model"], categories=model_order, ordered=True)
    combined_results = combined_results.sort_values("Model")

    # 确保列顺序正确
    combined_results = combined_results[column_order]

    # 打印表格到终端
    print(f"Results for Dataset: {dataset_name}\n")
    print(combined_results.to_string(index=False))  # 不打印索引
    print("\n")

    # 保存为 CSV 文件
    combined_results.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

    # 保存为 Excel 文件
    excel_filename = os.path.join(save_dir, f"{dataset_name}_results.xlsx")

    # 如果文件已存在，先删除
    if os.path.exists(excel_filename):
        os.remove(excel_filename)
        print(f"Deleted existing file: {excel_filename}")

    # 保存到 Excel
    combined_results.to_excel(excel_filename, index=False)
    print(f"Results saved to {excel_filename}")



# 主程序
if __name__ == "__main__":
    dataset_name = 'wine_quality'
    X, y = load_data(dataset_name)

    models = [
        SelfPacedEnsembleClassifier(n_jobs=-1),
        BalanceCascadeClassifier(n_jobs=-1),
        UnderBaggingClassifier(n_jobs=-1),
        EasyEnsembleClassifier(n_jobs=-1),
        RUSBoostClassifier(),
        BalancedRandomForestClassifier(n_jobs=-1),
        AdaCostClassifier(),
        AdaUBoostClassifier(),
        AsymBoostClassifier()
    ]

    # 是否使用布尔数组选择模型
    use_model_selection = False  # 如果为 True，则使用布尔数组；否则运行所有模型

    # 布尔数组，表示是否运行对应模型
    model_selection = [
        True,   # SelfPacedEnsembleClassifier
        True,  # BalanceCascadeClassifier
        True,   # UnderBaggingClassifier
        True,   # EasyEnsembleClassifier
        False,  # RUSBoostClassifier
        True,   # BalancedRandomForestClassifier
        False,  # AdaCostClassifier
        False,   # AdaUBoostClassifier
        False    # AsymBoostClassifier
    ]

    # 如果启用了模型选择功能，确保模型列表与布尔数组长度一致
    if use_model_selection:
        assert len(models) == len(model_selection), "Model selection array length must match the number of models."

    results = []  # 用于存储所有模型的结果

    # 遍历模型
    for i, model in enumerate(models):
        # 判断是否运行该模型
        if not use_model_selection or model_selection[i]:  # 如果未启用布尔数组或布尔值为 True
            model_name = model.__class__.__name__
            print(f"Running model: {model_name}")
            model_results = evaluate_and_save_predictions(model, X, y, dataset_name, model_name)
            # 添加模型名称，并按需要的指标顺序整理
            formatted_results = {
                "Model": model_name,  # 模型名称
                "acc": model_results["Accuracy"],
                "sen": model_results["Sensitivity"],
                "spe": model_results["Specificity"],
                "f1_macro": model_results["F1"],
                "gmean": model_results["G-mean"],
                "auc": model_results["AUC"],
                "aupr": model_results["AUPR"],
                "precision": model_results["Precision"],
                "recall": model_results["Recall"],
            }
            results.append(formatted_results)
        else:
            # 如果布尔值为 False，则跳过该模型
            print(f"Skipping model: {model.__class__.__name__}")

    # 保存和打印结果表格
    save_and_print_results_table(results, dataset_name)
