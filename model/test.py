import numpy as np
import matplotlib.pyplot as plt
from imbens.sampler import KMeansSMOTE
from sklearn.ensemble import RandomForestClassifier

# 模拟算法排名数据（平均排名和置信区间）
algorithms = [
    "ForSyn", "Original DF", "DCE-DForest", "DeepSynergy", "MatchMaker",
    "TranSynergy", "SynPathy", "XGBoost", "Random Forest", "RUS Boost", "Balanced Bagging"
]

# 示例数据：平均排名和置信区间（根据指标变化）
average_ranks = {
    "F1-score": [3, 4, 6, 8, 5, 7, 9, 10, 11, 12, 13],
    "AUPR": [2, 3, 5, 7, 6, 8, 9, 10, 11, 12, 13],
    "Recall": [1, 2, 4, 6, 5, 7, 9, 11, 10, 12, 13],
    "MCC": [2, 3, 6, 8, 5, 7, 10, 9, 11, 12, 13],
    "Gmean": [1, 3, 5, 7, 6, 8, 9, 10, 12, 11, 13]
}

# 假设置信区间宽度（以每个算法为例）
confidence_intervals = 0.5 * np.random.rand(len(algorithms), len(average_ranks))

# 绘制图表
fig, axs = plt.subplots(1, 5, figsize=(20, 6), sharey=True)

for i, (metric, ranks) in enumerate(average_ranks.items()):
    ax = axs[i]

    # 画出平均排名点和置信区间
    for j, (rank, ci) in enumerate(zip(ranks, confidence_intervals[:, i])):
        ax.errorbar(rank, j, xerr=ci, fmt='o', color='red', ecolor='blue', capsize=3)

    ax.set_title(f"Average Rank ({metric})")
    ax.set_xlim(-1, 14)  # 根据数据范围调整
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms if i == 0 else [])  # 仅第一列显示算法名称
    ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
