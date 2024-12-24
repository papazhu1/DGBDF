import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Set constants for plot styling
text_size = 30  # Adjust text size slightly to fit layout
linewidth = 2  # Line width for plot styling

# Generate an imbalanced dataset
X, y = make_classification(
    n_samples=300,  # Number of samples
    n_features=2,   # Using only 2 features to visualize the data
    n_informative=2,  # Number of informative features
    n_redundant=0,    # No redundant features
    n_classes=2,  # Binary classification
    weights=[0.8, 0.3],  # Imbalanced classes
    class_sep=2,         # Class separation
    random_state=42      # Ensures reproducibility
)

# Calculate the center of the minority class samples (this could be adjusted)
minority_class_samples = X[y == 1]
majority_class_samples = X[y == 0]
minority_center = np.mean(minority_class_samples, axis=0)

# Generate a few outliers near the minority class center
num_outliers = 50
outliers = minority_center + np.random.randn(num_outliers, 2) * 0.5  # Add small noise

# Add the outliers to the majority class
X = np.vstack([minority_class_samples, majority_class_samples, outliers])
y = np.hstack([y[y == 1], y[y == 0], np.zeros(num_outliers)])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a RandomForest classifier
rf = ExtraTreesClassifier(n_estimators=10, random_state=42, max_depth=6)
rf.fit(X_train, y_train)

# Collect evidence data
total_class_counts = []  # Store class-wise evidence for each test sample
S_values = []  # Store total evidence per test sample

for sample_idx, sample in enumerate(X_test):
    sample = sample.reshape(1, -1)  # Ensure sample has correct dimensions
    leaf_indices = rf.apply(sample).flatten()  # Get leaf indices for this sample
    total_samples_per_class = np.zeros(rf.n_classes_)  # Initialize class evidence counts

    for i, tree in enumerate(rf.estimators_):
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
negative_indices = np.where(y_test == 0)[0]
positive_indices = np.where(y_test == 1)[0]

# Select random indices for 20 majority class samples and 10 minority class samples
random_majority_indices = np.random.choice(negative_indices, 20, replace=False)
random_minority_indices = np.random.choice(positive_indices, 10, replace=False)

# Subset the data based on selected indices
selected_majority_class_counts = total_class_counts[random_majority_indices]
selected_minority_class_counts = total_class_counts[random_minority_indices]
majority_counts = selected_majority_class_counts[:, 0]
minority_counts = selected_majority_class_counts[:, 1]
overlap_majority = np.array([min(maj, mino) for maj, mino in zip(majority_counts, minority_counts)])  # Calculate overlap
max_value_majority = np.array([max(maj, mino) for maj, mino in zip(majority_counts, minority_counts)])  # Calculate max value
blue_part_majority = [maj - ov for maj, ov in zip(majority_counts, overlap_majority)]
orange_part_majority = [mino - ov for mino, ov in zip(minority_counts, overlap_majority)]

majority_sorted_indices = np.argsort(max_value_majority)[::-1]  # Sort in descending order
sorted_blue_part_majority = np.array(blue_part_majority)[majority_sorted_indices]
sorted_orange_part_majority = np.array(orange_part_majority)[majority_sorted_indices]
sorted_overlap_majority = overlap_majority[majority_sorted_indices]

selected_minority_class_counts = total_class_counts[random_minority_indices]
majority_counts_min = selected_minority_class_counts[:, 0]
minority_counts_min = selected_minority_class_counts[:, 1]
overlap_minority = np.array([min(maj, mino) for maj, mino in zip(majority_counts_min, minority_counts_min)])  # Calculate overlap
max_value_minority = np.array([max(maj, mino) for maj, mino in zip(majority_counts_min, minority_counts_min)])  # Calculate max value
blue_part_minority = [maj - ov for maj, ov in zip(majority_counts_min, overlap_minority)]
orange_part_minority = [mino - ov for mino, ov in zip(minority_counts_min, overlap_minority)]

minority_sorted_indices = np.argsort(max_value_minority)[::-1]  # Sort in descending order
sorted_blue_part_minority = np.array(blue_part_minority)[minority_sorted_indices]
sorted_orange_part_minority = np.array(orange_part_minority)[minority_sorted_indices]
sorted_overlap_minority = overlap_minority[minority_sorted_indices]

selected_S_values_log_majority = S_values_log[random_majority_indices]
selected_S_values_log_minority = S_values_log[random_minority_indices]
selected_labels_majority = y_test[random_majority_indices]
selected_labels_minority = y_test[random_minority_indices]

# Define colors
majority_color = '#8ab1ef'
minority_color = '#ffcc95'
mixed_color = "#97a1a5"  # 混合色

# Create subplots with larger figsize
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Subplot 1: Improved Bar chart for majority class evidence
x_majority = np.arange(len(sorted_blue_part_majority))

axes[0].bar(
    x_majority, sorted_blue_part_majority, width=1, color=majority_color, edgecolor='black', linewidth=2.5,
    bottom=sorted_overlap_majority, label='Majority ($c_2$)', alpha=0.85
)
axes[0].bar(
    x_majority, sorted_orange_part_majority, width=1, color=minority_color, edgecolor='black', linewidth=2.5,
    bottom=sorted_overlap_majority, label='Minority ($c_1$)', alpha=1
)
axes[0].bar(
    x_majority, sorted_overlap_majority, width=1, color=mixed_color, edgecolor='black', linewidth=2.5,
    label='Overlap', alpha=0.9
)

# Set y-axis limit to avoid exceeding plot bounds
max_evidence_majority = selected_majority_class_counts.sum(axis=1).max()  # Calculate maximum bar height
axes[0].set_ylim(0, max_evidence_majority * 1.1)  # Set ylim slightly above the maximum evidence

# Enhance axis labels and title
axes[0].set_xlabel("Majority Class Test Samples", fontsize=text_size)
axes[0].set_ylabel("Evidence Sum", fontsize=text_size)
axes[0].set_title("Majority Class Evidence Distribution", fontsize=text_size)
axes[0].legend(fontsize=text_size, loc='upper right')

# Add grid and improve readability
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0].tick_params(axis='both', which='major', labelsize=text_size)
axes[0].tick_params(axis='both', which='minor', labelsize=text_size)

# Subplot 2: Improved Bar chart for minority class evidence
x_minority = np.arange(len(sorted_blue_part_minority))

axes[1].bar(
    x_minority, sorted_blue_part_minority, width=1, color=majority_color, edgecolor='black', linewidth=2.5,
    bottom=sorted_overlap_minority, label='Majority ($c_2$)', alpha=0.85
)
axes[1].bar(
    x_minority, sorted_orange_part_minority, width=1, color=minority_color, edgecolor='black', linewidth=2.5,
    bottom=sorted_overlap_minority, label='Minority ($c_1$)', alpha=1
)
axes[1].bar(
    x_minority, sorted_overlap_minority, width=1, color=mixed_color, edgecolor='black', linewidth=2.5,
    label='Overlap', alpha=0.9
)

# Set y-axis limit to avoid exceeding plot bounds
max_evidence_minority = selected_minority_class_counts.sum(axis=1).max()  # Calculate maximum bar height
axes[1].set_ylim(0, max_evidence_minority * 1.1)  # Set ylim slightly above the maximum evidence

# Enhance axis labels and title
axes[1].set_xlabel("Minority Class Test Samples", fontsize=text_size)
axes[1].set_ylabel("Evidence Sum", fontsize=text_size)
# Subplot 2 (cont'd): Improved Bar chart for minority class evidence
axes[1].set_title("Minority Class Evidence Distribution", fontsize=text_size)
axes[1].legend(fontsize=text_size, loc='upper right')

# Add grid and improve readability
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1].tick_params(axis='both', which='major', labelsize=text_size)
axes[1].tick_params(axis='both', which='minor', labelsize=text_size)

# Adjust layout and show plots
plt.tight_layout()
plt.show()

