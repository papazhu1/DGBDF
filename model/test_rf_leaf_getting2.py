import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imbens.ensemble.under_sampling import RUSBoostClassifier, SelfPacedEnsembleClassifier, BalanceCascadeClassifier, \
    EasyEnsembleClassifier
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
import joblib  # 用于保存模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


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

    minority_center = np.mean(minority_class_samples, axis=0)

    # Generate a few outliers near the minority class center
    num_outliers = 0
    outliers = minority_center + np.random.randn(num_outliers, 2) * 0.5  # Add small noise

    # Add the outliers to the majority class
    X = np.vstack([minority_class_samples, majority_class_samples, outliers])
    y = np.hstack([y[y == 1], y[y == 0], np.zeros(num_outliers)])

    # Train a RandomForest classifier
    rf = EasyEnsembleClassifier(n_estimators=20)
    rf.fit(X, y)

    # 保存训练好的模型
    joblib.dump(rf, "random_forest_model.pkl")
    print("Model saved as 'random_forest_model.pkl'.")

    # Collect evidence data
    total_class_counts = []  # Store class-wise evidence for each sample
    S_values = []  # Store total evidence per sample

    for sample_idx, sample in enumerate(X):
        sample = sample.reshape(1, -1)  # Ensure sample has correct dimensions
        # leaf_indices = rf.apply(sample).flatten()  # Get leaf indices for this sample
        leaf_indices = []  # 存储该样本的叶子索引

        # 遍历 EasyEnsembleClassifier 的每个基分类器（AdaBoost 或 Pipeline）
        for ensemble_idx, adaboost in enumerate(rf.estimators_):
            # 检查是否是 Pipeline
            if isinstance(adaboost, Pipeline):
                # 如果是 Pipeline，提取其中的实际分类器
                adaboost = adaboost.steps[-1][1]  # 提取 Pipeline 的最后一步

            # 遍历基分类器中的弱分类器（决策树）
            for tree_idx, tree in enumerate(adaboost.estimators_):
                if isinstance(tree, DecisionTreeClassifier):  # 确保是决策树
                    # 获取该样本在树中的叶子索引
                    leaf_index = tree.apply(sample)
                    leaf_indices.append(leaf_index[0])  # 存储叶子索引

        total_samples_per_class = np.zeros(rf.n_classes_)  # Initialize class evidence counts

        for i, adaboost in enumerate(rf.estimators_):
            if isinstance(adaboost, Pipeline):
                adaboost = adaboost.steps[-1][1]  # 提取 Pipeline 的最后一步

            for tree_idx, tree in enumerate(adaboost.estimators_):
                if isinstance(tree, DecisionTreeClassifier):
                    tree_structure = tree.tree_
                    leaf_index = leaf_indices[i]  # Get the leaf index for this tree
                    samples_per_class = tree_structure.value[leaf_index, 0]  # Class evidence in this leaf
                    total_samples_per_class += samples_per_class  # Accumulate evidence counts

        S = np.sum(total_samples_per_class)  # Total evidence for this sample
        total_class_counts.append(total_samples_per_class)
        S_values.append(S)

    # Convert to NumPy arrays
    total_class_counts = np.array(total_class_counts)
    S_values = np.array(S_values)

    # Compute log-scaled evidence
    S_values_log = np.log1p(S_values)  # log(1 + S)

    # Separate indices for negative and positive classes
    negative_indices = np.where(y == 0)[0]
    positive_indices = np.where(y == 1)[0]

    # Select random indices for 20 majority class samples and 10 minority class samples
    random_majority_indices = np.random.choice(negative_indices, 20, replace=False)
    random_minority_indices = np.random.choice(positive_indices, 10, replace=False)

    # Subset the data based on selected indices
    selected_majority_class_counts = total_class_counts[random_majority_indices]
    selected_minority_class_counts = total_class_counts[random_minority_indices]
    majority_counts = selected_majority_class_counts[:, 0]
    minority_counts = selected_majority_class_counts[:, 1]
    overlap_majority = np.array([min(maj, mino) for maj, mino in zip(majority_counts, minority_counts)])
    max_value_majority = np.array([max(maj, mino) for maj, mino in zip(majority_counts, minority_counts)])
    blue_part_majority = [maj - ov for maj, ov in zip(majority_counts, overlap_majority)]
    orange_part_majority = [mino - ov for mino, ov in zip(minority_counts, overlap_majority)]

    majority_sorted_indices = np.argsort(max_value_majority)[::-1]  # Sort in descending order
    sorted_blue_part_majority = np.array(blue_part_majority)[majority_sorted_indices]
    sorted_orange_part_majority = np.array(orange_part_majority)[majority_sorted_indices]
    sorted_overlap_majority = overlap_majority[majority_sorted_indices]

    selected_minority_class_counts = total_class_counts[random_minority_indices]
    majority_counts_min = selected_minority_class_counts[:, 0]
    minority_counts_min = selected_minority_class_counts[:, 1]
    overlap_minority = np.array([min(maj, mino) for maj, mino in zip(majority_counts_min, minority_counts_min)])
    max_value_minority = np.array([max(maj, mino) for maj, mino in zip(majority_counts_min, minority_counts_min)])
    blue_part_minority = [maj - ov for maj, ov in zip(majority_counts_min, overlap_minority)]
    orange_part_minority = [mino - ov for mino, ov in zip(minority_counts_min, overlap_minority)]

    minority_sorted_indices = np.argsort(max_value_minority)[::-1]  # Sort in descending order
    sorted_blue_part_minority = np.array(blue_part_minority)[minority_sorted_indices]
    sorted_orange_part_minority = np.array(orange_part_minority)[minority_sorted_indices]
    sorted_overlap_minority = overlap_minority[minority_sorted_indices]

    # 保存选中的样本数据
    selected_samples = np.vstack([X[random_majority_indices], X[random_minority_indices]])
    selected_labels = np.hstack([y[random_majority_indices], y[random_minority_indices]])
    selected_predictions = rf.predict(selected_samples)  # 模型的预测结果

    # 将数据保存为 CSV 文件
    data_to_save = np.hstack([selected_samples, selected_labels.reshape(-1, 1), selected_predictions.reshape(-1, 1)])
    columns = ["Feature 1", "Feature 2", "True Label", "Predicted Label"]
    df = pd.DataFrame(data_to_save, columns=columns)
    df.to_csv("selected_samples_predictions.csv", index=False)

    # Print confirmation of data saving
    print("Selected samples and predictions saved as 'selected_samples_predictions.csv'.")

    # 创建网格用于绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # 使用训练好的模型预测网格点
    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘图代码...
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)  # 添加决策边界
    # Subplot 1: Visualization of selected samples
    axes[0].scatter(X[random_majority_indices, 0], X[random_majority_indices, 1], color='navy', label="Majority", alpha=0.6, s=150)
    axes[0].scatter(X[random_minority_indices, 0], X[random_minority_indices, 1], color='orangered', label="Minority", alpha=0.6, s=150)
    axes[0].set_xlabel("Feature 1", fontsize=text_size)
    axes[0].set_ylabel("Feature 2", fontsize=text_size)
    axes[0].legend(fontsize=text_size, loc='upper right')
    axes[0].grid(True, linestyle="dotted", alpha=0.7, linewidth=2.5)

    # Subplot 2: Majority Class Evidence
    x_majority = np.arange(len(sorted_blue_part_majority))
    axes[1].bar(
        x_majority, sorted_blue_part_majority, width=1, color='#8ab1ef', edgecolor='black', linewidth=2.5,
        bottom=sorted_overlap_majority, label='Majority', alpha=0.85
    )
    axes[1].bar(
        x_majority, sorted_orange_part_majority, width=1, color='#ffcc95', edgecolor='black', linewidth=2.5,
        bottom=sorted_overlap_majority, label='Minority', alpha=1
    )
    axes[1].bar(
        x_majority, sorted_overlap_majority, width=1, color="#97a1a5", edgecolor='black', linewidth=2.5,
        label='Overlap', alpha=0.9
    )
    axes[1].set_ylim(0, max(sorted_blue_part_majority + sorted_orange_part_majority + sorted_overlap_majority) * 1.5)
    axes[1].set_xlabel("Majority Class Instances", fontsize=text_size)
    axes[1].set_ylabel("Evidence Sum", fontsize=text_size)
    axes[1].legend(fontsize=text_size, loc='upper right')
    axes[1].grid(axis='y', linestyle='dotted', alpha=0.7, linewidth=3.0)

    # Subplot 3: Minority Class Evidence
    x_minority = np.arange(len(sorted_blue_part_minority))
    axes[2].bar(
        x_minority, sorted_blue_part_minority, width=1, color='#8ab1ef', edgecolor='black', linewidth=2.5,
        bottom=sorted_overlap_minority, label='Majority', alpha=0.85
    )
    axes[2].bar(
        x_minority, sorted_orange_part_minority, width=1, color='#ffcc95', edgecolor='black', linewidth=2.5,
        bottom=sorted_overlap_minority, label='Minority', alpha=1
    )
    axes[2].bar(
        x_minority, sorted_overlap_minority, width=1, color="#97a1a5", edgecolor='black', linewidth=2.5,
        label='Overlap', alpha=0.9
    )
    axes[2].set_ylim(0, max(sorted_blue_part_minority + sorted_orange_part_minority + sorted_overlap_minority) * 1.5)
    axes[2].set_xlabel("Minority Class Instances", fontsize=text_size)
    axes[2].set_ylabel("Evidence Sum", fontsize=text_size)
    axes[2].legend(fontsize=text_size, loc='upper right')
    axes[2].grid(axis='y', linestyle='dotted', alpha=0.7, linewidth=3.0)

    # 加粗所有子图的边框
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)

    # 调整x轴和y轴刻度标签的字体大小
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=12)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("fig/selected_samples_and_evidence2.jpg")
    plt.show()

