import pandas as pd
from scipy import stats
import numpy as np
from statsmodels.stats.multitest import multipletests

# Load the dataset from the provided Excel file
file_path = r'C:\Users\10928\Documents\GitHub\DGBDF\model\compared_results\f1-macro_xlsx.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Function to clean performance values and keep only the mean
def clean_performance(value):
    try:
        value = str(value).split(' $\pm$ ')[0]  # Keep only the part before '±'
        return float(value)
    except ValueError:
        return np.nan  # Return NaN if the value cannot be converted

# Apply the cleaning function to each performance column
for column in df.columns[1:]:  # Exclude the first column 'Dataset'
    df[column] = df[column].apply(clean_performance)
df = df[1:]
# Display the cleaned dataframe
print("Cleaned DataFrame:")
print(df.head())


# Define performance columns (all columns except 'Dataset')
performance_columns = [col for col in df.columns if col != 'Dataset']

# Convert all performance columns to numeric values, forcing errors to NaN (useful for missing or incorrect data)
df[performance_columns] = df[performance_columns].apply(pd.to_numeric, errors='coerce')

print(df)
# Handle any missing (NaN) values by either filling or removing them (here we'll remove rows with NaN values)
df.dropna(subset=performance_columns, inplace=True)

print(df)
# Step 1: Rank each method (classifier) for each dataset
ranks_df = df[performance_columns].rank(axis=1, method='average', ascending=False)

# Step 2: Compute the average rank for each method
average_ranks = ranks_df.mean(axis=0)

# Display average ranks
print("\nAverage ranks for each method:")
print(average_ranks)

performance_columns = ranks_df
# Step 3: Perform the Friedman test across the classifiers
friedman_results = stats.friedmanchisquare(*[df[col] for col in performance_columns])

# Step 4: Check for significance in the Friedman test
print("Friedman Test p-value:", friedman_results.pvalue)

print("performance_columns", performance_columns)
if friedman_results.pvalue < 0.05:
    print("There are significant differences between the classifiers (Friedman test).")

    # Step 5: Perform pairwise comparisons (Mann-Whitney U test as an example)
    p_values = []
    for i in range(len(performance_columns)):
        for j in range(i + 1, len(performance_columns)):
            # print(df[performance_columns[i]])
            # print(df[performance_columns[j]])
            _, p_val = stats.mannwhitneyu(df[performance_columns[i]], df[performance_columns[j]], alternative='two-sided')

            # group1 = ranks_df[performance_columns[i]]
            # group2 = ranks_df[performance_columns[j]]
            #
            # _, p_val = stats.kruskal(group1, group2)
            if j == len(performance_columns) - 1:
                print(f"{p_val:.8f}")
            p_values.append((i, j, p_val))  # Store the indices and p-value as a tuple

    # Step 6: Apply Holm correction to adjust p-values for multiple comparisons
    _, corrected_p_values, _, _ = multipletests([p[2] for p in p_values], alpha=0.05, method='holm')

    print("Post Hoc P-values after Holm correction:")
    for idx, p_val in enumerate(corrected_p_values):
        i, j, _ = p_values[idx]  # Extract indices from the tuple
        if j == 8:
            print(f"Comparison between {performance_columns[i]} and {performance_columns[j]}: p-value = {p_val:.8f}")

else:
    print("No significant differences detected by the Friedman test.")

# # Step 7: Perform Kruskal-Wallis test (alternative to ANOVA)
# kruskal_results = stats.kruskal(*[df[col] for col in performance_columns])

# # Step 8: Check for significance in the Kruskal-Wallis test
# print("Kruskal-Wallis Test p-value:", kruskal_results.pvalue)
#
# if kruskal_results.pvalue < 0.05:
#     print("Significant differences detected by the Kruskal-Wallis test.")
# else:
#     print("No significant differences detected by the Kruskal-Wallis test.")
