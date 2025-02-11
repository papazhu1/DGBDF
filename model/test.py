import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.ensemble import EasyEnsembleClassifier
from imbens.ensemble import *
from sklearn.ensemble import RandomForestClassifier
import joblib  # 用于保存模型
import json  # 用于保存种子
from sklearn.datasets import make_moons
from collections import Counter
from matplotlib.lines import Line2D
from UADF import UncertaintyAwareDeepForest
from demo import get_config
from imbens.metrics import geometric_mean_score
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from deepforest import CascadeForestClassifier

def interpolate_safe_majority_class(X, y, classifier, threshold=0.5, num_samples_to_generate=20, seed=42):
    """
    对安全的多数类样本进行插值生成新样本，并保存插值的种子
    X: 数据特征
    y: 类别标签
    classifier: 已训练的分类器
    threshold: 确定“安全样本”的预测概率阈值
    num_samples_to_generate: 需要插值生成的样本数
    seed: 随机种子
    """
    # 设置随机种子，以确保可重复性
    # np.random.seed(seed)
    
    # 计算每个样本的预测概率
    proba = classifier.predict_proba(X)
    
    # 选择安全样本：对于多数类（类别0），其属于类别0的概率大于阈值
    majority_class_samples = X[y == 0]
    majority_class_probs = proba[y == 0, 0]  # 获取多数类样本属于类别0的概率
    majority_class_probs = np.array(majority_class_probs)
    
    print(majority_class_probs)
    print(np.where(majority_class_probs > threshold))
    # 选择概率大于阈值的安全样本
    safe_samples_idx = np.where(majority_class_probs > threshold)[0]
    print(safe_samples_idx)
    safe_samples = majority_class_samples[safe_samples_idx]

    # 用于存储插值生成的新样本
    new_samples = []
    
    # 插值过程：在安全样本之间进行插值
    for _ in range(num_samples_to_generate):
        # 随机选择两个安全多数类样本
        idx1, idx2 = np.random.choice(len(safe_samples), 2, replace=False)
        sample1, sample2 = safe_samples[idx1], safe_samples[idx2]
        
        # 在这两个样本之间进行线性插值
        new_sample = sample1 + np.random.rand() * (sample2 - sample1)
        
        # 将插值生成的新样本添加到列表中
        new_samples.append(new_sample)
    
    # 将生成的新样本添加到原数据中
    new_samples = np.array(new_samples)
    X_new = np.vstack([X, new_samples])
    y_new = np.hstack([y, np.zeros(num_samples_to_generate)])  # 新样本的标签为0（多数类）
    
    # 保存随机种子到文件
    with open("interpolation_seed.json", "w") as f:
        json.dump({"seed": seed}, f)
    
    return X_new, y_new

def evaluate_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # 获取预测正类的概率

    f1_macro = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_prob)
    aupr = average_precision_score(y_test, y_prob)
    gmean = geometric_mean_score(y_test, y_pred)

    return {
        "F1-macro": f1_macro,
        "AUC": auc,
        "AUPR": aupr,
        "Gmean": gmean
    }

# 循环10次并计算平均值和标准差
def evaluate_multiple_runs(model_class, model_params, X, y, n_runs=10):
    results = {"F1-macro": [], "AUC": [], "AUPR": [], "Gmean": []}

    for i in range(n_runs):
        
        # 训练EasyEnsemble分类器
        rf = EasyEnsembleClassifier(n_estimators=20)
        rf.fit(X, y)
        
        # 对安全多数类样本进行插值
        X_train, y_train = interpolate_safe_majority_class(X, y, rf, threshold=0.6, num_samples_to_generate=80, seed=5)
        
        X_test, y_test = make_moons(n_samples=10, noise=0.3, random_state=37)
        X_test, y_test = interpolate_safe_majority_class(X_test, y_test, rf, threshold=0.6, num_samples_to_generate=45, seed=5)
        
        # 初始化并训练模型
        model = model_class(**model_params)
        # model = UncertaintyAwareDeepForest(config=get_config())
        model.fit(X_train, y_train)
        

        # 评估模型
        performance = evaluate_model_performance(model, X_test, y_test)

        # 记录结果
        for metric, value in performance.items():
            results[metric].append(value)

    # 计算平均值和标准差
    avg_std_results = {
        metric: {
            "mean": np.mean(values),
            "std": np.std(values)
        }
        for metric, values in results.items()
    }

    return avg_std_results

if __name__ == "__main__":
    # Set constants for plot styling
    text_size = 30  # Adjust text size slightly to fit layout
    linewidth = 2  # Line width for plot styling

    # 生成均衡数据集
    X, y = make_moons(n_samples=40, noise=0.3, random_state=36)

    # 对类别 1 进行下采样
    minority_class_samples = X[y == 1]
    majority_class_samples = X[y == 0]

    # 保留所有的多数类样本，减少少数类样本至 50 个
    minority_class_samples = minority_class_samples[:10]

    # 合并数据，生成不均衡数据集
    X = np.vstack([majority_class_samples, minority_class_samples])
    y = np.hstack([np.zeros(len(majority_class_samples)), np.ones(len(minority_class_samples))])

    # 训练EasyEnsemble分类器
    rf = EasyEnsembleClassifier(n_estimators=20)
    rf.fit(X, y)
    
    # 对安全多数类样本进行插值
    X_resampled, y_resampled = interpolate_safe_majority_class(X, y, rf, threshold=0.6, num_samples_to_generate=80, seed=5)
    
    X_test, y_test = make_moons(n_samples=10, noise=0.3, random_state=37)
    X_test_resampled, y_test_resampled = interpolate_safe_majority_class(X, y, rf, threshold=0.6, num_samples_to_generate=45, seed=5)
    
    models_to_evaluate = [
    {"name": "EasyEnsembleClassifier", "class": EasyEnsembleClassifier, "params": {"n_estimators": 100}, "X_train": X, "y_train": y},
    {"name": "UnderBagging", "class": UnderBaggingClassifier, "params": {"n_estimators": 100}, "X_train": X, "y_train": y},
    {"name": "BalancedRandomForest", "class": BalancedRandomForestClassifier, "params": {"n_estimators": 100}, "X_train": X, "y_train": y},
    {"name": "BalanceCascadeClassifier", "class": BalanceCascadeClassifier, "params": {"n_estimators": 100}, "X_train": X, "y_train": y},
    {"name": "RUSBoost", "class": RUSBoostClassifier, "params": {"n_estimators": 100}, "X_train": X, "y_train": y},
    {"name": "SelfPacedEnsemble", "class": SelfPacedEnsembleClassifier, "params": {"n_estimators": 100}, "X_train": X, "y_train": y}
    ]

    # 初始化表格数据
    results_table = []
    
    # 循环评估多个模型
    for model_info in models_to_evaluate:
        print(f"Evaluating {model_info['name']}:")
        results = evaluate_multiple_runs(
            model_class=model_info["class"],
            model_params=model_info["params"],
            X=model_info["X_train"],
            y=model_info["y_train"]
        )
        
        # 整理结果到表格
        results_table.append({
            "Model": model_info["name"],
            **{metric: f"{values['mean']:.4f} ± {values['std']:.4f}" for metric, values in results.items()}
        })
    
    # 转换为DataFrame并打印
    results_df = pd.DataFrame(results_table)
    print(results_df)
    
    # 保存结果到CSV文件
    results_df.to_csv("artificial_dataset_performance_results.csv", index=False)

    # 读取保存的 resampled 数据
    # X_resampled = np.load("X_resampled.npy")
    # y_resampled = np.load("y_resampled.npy")

    print(Counter(y_resampled))
    
    ee = EasyEnsembleClassifier(n_estimators=100)
    ee.fit(X_resampled, y_resampled)
    
    evaluate_model_performance(ee, X_test_resampled, y_test_resampled)
    
    # 训练BalancedCascade分类器
    bc = BalanceCascadeClassifier(n_estimators=100)
    bc.fit(X, y)
    
    evaluate_model_performance(bc, X_test_resampled, y_test_resampled)

    # 创建网格用于绘制决策边界
    x_min, x_max = X_resampled[:, 0].min() - 1, X_resampled[:, 0].max() + 1
    y_min, y_max = X_resampled[:, 1].min() - 1, X_resampled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    # 使用EasyEnsemble模型预测网格点
    Z_easyensemble = ee.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_easyensemble = Z_easyensemble.reshape(xx.shape)
    
    
    # 使用BalancedCascade模型预测网格点
    Z_balancedcascade = bc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_balancedcascade = Z_balancedcascade.reshape(xx.shape)
    
    # 创建子图并设置大小
    fig, ax = plt.subplots(figsize=(8,8), dpi=400)
    # 绘制样本点
    scatter = ax.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, cmap=plt.cm.Paired, s=150, edgecolor='black', alpha=0.4)
    # 绘制EasyEnsemble决策边界（只绘制虚线边界）
    ax.contour(xx, yy, Z_easyensemble, colors='blue', linewidths=2, linestyles='dashed')
    ax.contour(xx, yy, Z_balancedcascade, colors='green', linewidths=2, linestyles='dashed')
    plt.show()
    
    uadf = UncertaintyAwareDeepForest(config=get_config())
    uadf.fit(X_resampled, y_resampled)
    
    evaluate_model_performance(uadf, X_test_resampled, y_test_resampled)
    
    
    # 使用UADF模型预测网格点
    Z_uadf = uadf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_uadf = Z_uadf.reshape(xx.shape)

    # 创建子图并设置大小
    fig, ax = plt.subplots(figsize=(8,8), dpi=400)

    # 绘制EasyEnsemble决策边界（只绘制虚线边界）
    ax.contour(xx, yy, Z_easyensemble, colors='blue', linewidths=2, linestyles='dashed')  # EasyEnsemble: 虚线边界
    
    # 绘制BalancedCascade决策边界（只绘制虚线边界）
    ax.contour(xx, yy, Z_balancedcascade, colors='green', linewidths=2, linestyles='dashed')  # BalancedCascade: 虚线边界

    ax.contour(xx, yy, Z_uadf, colors='purple', linewidths=2, linestyles='dashed')
    
    
    
    # 绘制样本点
    scatter = ax.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, cmap=plt.cm.Paired, s=150, edgecolor='black', alpha=0.6)

    # 手动设置legend
    handles, labels = scatter.legend_elements()
    labels = ['Majority', 'Minority']  # 修改legend标签

    # Create custom legend for decision boundaries
    boundary_legend = [Line2D([0], [0], color='blue', lw=2, label='EasyEnsemble', linestyle='dashed'),
                       Line2D([0], [0], color='green', lw=2, label='BalancedCascade', linestyle='dashed'),
                       Line2D([0], [0], color='purple', lw=2, label='UABDF', linestyle='dashed')]
    
    class_patch = Line2D([0], [0], color='white', lw=0, label='Class')
    # Combine both legends
    ax.legend([class_patch] + handles + [class_patch] + boundary_legend, ['class'] + labels + ['decision boundary'] + ['EasyEnsemble', 'BalancedCascade', 'UABDF'], fontsize=12, loc='upper right')

    ax.set_xlabel("Feature 1", fontsize=text_size)
    ax.set_ylabel("Feature 2", fontsize=text_size)
    ax.grid(True, linestyle="dotted", alpha=0.7, linewidth=2.5)

    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    # 调整x轴和y轴刻度标签的字体大小
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig("fig/decision_boundaries_both_models_with_legend.jpg")
    plt.show()
