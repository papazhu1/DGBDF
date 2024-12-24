import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Set constants for plot styling
text_size = 30  # Adjust text size slightly to fit layout
linewidth = 2  # Line width for plot styling

# 读取保存的模型
model = joblib.load("random_forest_model.pkl")

# 读取保存的样本数据和预测结果
data = pd.read_csv("selected_samples_predictions.csv")

# 提取特征、真实标签和预测标签
X = data[["Feature 1", "Feature 2"]].values
true_labels = data["True Label"].values
predicted_labels = data["Predicted Label"].values

# Separate indices for majority and minority classes based on true labels
majority_indices = np.where(true_labels == 0)[0]
minority_indices = np.where(true_labels == 1)[0]

# Collect evidence data
total_class_counts = []  # Store class-wise evidence for each sample
S_values = []  # Store total evidence per sample

for sample_idx, sample in enumerate(X):
    sample = sample.reshape(1, -1)  # Ensure sample has correct dimensions
    leaf_indices = model.apply(sample).flatten()  # Get leaf indices for this sample
    total_samples_per_class = np.zeros(model.n_classes_)  # Initialize class evidence counts

    for i, tree in enumerate(model.estimators_):
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

# Subset the data for visualization
selected_majority_class_counts = total_class_counts[majority_indices][:20]
selected_minority_class_counts = total_class_counts[minority_indices][:10]

majority_counts = selected_majority_class_counts[:, 0]
minority_counts = selected_majority_class_counts[:, 1]
overlap_majority = np.minimum(majority_counts, minority_counts)

majority_sorted_indices = np.argsort(np.maximum(majority_counts, minority_counts))[::-1]
sorted_blue_part_majority = (majority_counts - overlap_majority)[majority_sorted_indices]
sorted_orange_part_majority = (minority_counts - overlap_majority)[majority_sorted_indices]
sorted_overlap_majority = overlap_majority[majority_sorted_indices]

minority_counts_min = selected_minority_class_counts[:, 1]
majority_counts_min = selected_minority_class_counts[:, 0]
overlap_minority = np.minimum(minority_counts_min, majority_counts_min)

minority_sorted_indices = np.argsort(np.maximum(minority_counts_min, majority_counts_min))[::-1]
sorted_blue_part_minority = (majority_counts_min - overlap_minority)[minority_sorted_indices]
sorted_orange_part_minority = (minority_counts_min - overlap_minority)[minority_sorted_indices]
sorted_overlap_minority = overlap_minority[minority_sorted_indices]

# Define colors
majority_color = '#8ab1ef'
minority_color = '#ffcc95'
mixed_color = "#97a1a5"

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Subplot 1: Visualization of selected samples
axes[0].scatter(X[majority_indices, 0], X[majority_indices, 1], color=majority_color, label="Majority", alpha=0.6, s=100)
axes[0].scatter(X[minority_indices, 0], X[minority_indices, 1], color=minority_color, label="Minority", alpha=0.6, s=100)
axes[0].set_xlabel("Feature 1", fontsize=text_size)
axes[0].set_ylabel("Feature 2", fontsize=text_size)
axes[0].legend(fontsize=text_size, loc='upper right')
axes[0].grid(True, linestyle="dotted", alpha=0.7, linewidth=2.5)

# Subplot 2: Majority Class Evidence
x_majority = np.arange(len(sorted_blue_part_majority))
axes[1].bar(
    x_majority, sorted_blue_part_majority, width=1, color=majority_color, edgecolor='black', linewidth=2.5,
    bottom=sorted_overlap_majority, label='Majority', alpha=0.85
)
axes[1].bar(
    x_majority, sorted_orange_part_majority, width=1, color=minority_color, edgecolor='black', linewidth=2.5,
    bottom=sorted_overlap_majority, label='Minority', alpha=1
)
axes[1].bar(
    x_majority, sorted_overlap_majority, width=1, color=mixed_color, edgecolor='black', linewidth=2.5,
    label='Overlap', alpha=0.9
)
axes[1].set_ylim(0, max(sorted_blue_part_majority + sorted_orange_part_majority + sorted_overlap_majority) * 1.2)
axes[1].set_xlabel("Majority Class Instances", fontsize=text_size)
axes[1].set_ylabel("Evidence Sum", fontsize=text_size)
axes[1].legend(fontsize=text_size, loc='upper right')
axes[1].grid(axis='y', linestyle='dotted', alpha=0.7, linewidth=3.0)

# Subplot 3: Minority Class Evidence
x_minority = np.arange(len(sorted_blue_part_minority))
axes[2].bar(
    x_minority, sorted_blue_part_minority, width=1, color=majority_color, edgecolor='black', linewidth=2.5,
    bottom=sorted_overlap_minority, label='Majority', alpha=0.85
)
axes[2].bar(
    x_minority, sorted_orange_part_minority, width=1, color=minority_color, edgecolor='black', linewidth=2.5,
    bottom=sorted_overlap_minority, label='Minority', alpha=1
)
axes[2].bar(
    x_minority, sorted_overlap_minority, width=1, color=mixed_color, edgecolor='black', linewidth=2.5,
    label='Overlap', alpha=0.9
)
axes[2].set_ylim(0, max(sorted_blue_part_minority + sorted_orange_part_minority + sorted_overlap_minority) * 1.2)
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
plt.savefig("fig/visualized_from_stored_data.jpg")
plt.show()
