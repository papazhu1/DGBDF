import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

# 生成均衡数据集
X, y = make_moons(n_samples=40, noise=0.3, random_state=36)

# 对类别 1 进行下采样
minority_class_samples = X[y == 1]
majority_class_samples = X[y == 0]

# 保留所有的多数类样本，减少少数类样本至 10 个
minority_class_samples = minority_class_samples[:10]

# 合并数据，生成不均衡数据集
X = np.vstack([majority_class_samples, minority_class_samples])
y = np.hstack([np.zeros(len(majority_class_samples)), np.ones(len(minority_class_samples))])

# 特殊点的索引
special_indices = [7, 3, 15, 16]

# 可视化数据
plt.figure(figsize=(6, 6), dpi=300)

# 绘制大多数类别样本
plt.scatter(majority_class_samples[:, 0], majority_class_samples[:, 1],
            color='blue', s=300, label='Majority Class', alpha=0.6)

# 绘制少数类别样本
plt.scatter(minority_class_samples[:, 0], minority_class_samples[:, 1],
            color='red', s=300, label='Minority Class', alpha=0.6)

# 绘制特殊点的灰色背景，增加外环宽度
for idx in special_indices:
    plt.scatter(X[idx, 0], X[idx, 1], color='gray', s=700, alpha=0.5, zorder=1)  # 增加 s 参数的值


# 绘制特殊点
plt.scatter(X[special_indices, 0], X[special_indices, 1],
            color='blue', s=300, edgecolor='none', zorder=2)

# 绘制虚线边框更贴近数据点
for idx in special_indices:
    plt.scatter(X[idx, 0], X[idx, 1], facecolors='none', edgecolors='black',
                s=700, linewidth=2, linestyle='dotted', alpha=0.9, zorder=3)

# 标注每个点的索引
for i, txt in enumerate(range(len(X))):
    plt.annotate(txt, (X[i, 0], X[i, 1]), fontsize=8, alpha=0.7, ha='center', va='center', color='black')

# 创建4个分类器
clf1 = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf3 = ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf4 = ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1)

# 集成模型（投票分类器）
ensemble_clf = VotingClassifier(
    estimators=[
        ('rf1', clf1),
        ('rf2', clf2),
        ('et1', clf3),
        ('et2', clf4),
    ],
    voting='soft'  # 使用软投票（概率加权平均）
)

# 设置图例
plt.legend()

# 设置标题和标签
plt.title('Scatter plot with adjusted dashed borders for special points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.show()
