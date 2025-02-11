import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib.pyplot as plt

from model.compared_results.std_deviation_wilcoxon import compute_cd

def compute_critical_difference(avranks, names, cd, width=10):
    """
    Draws Critical Difference (CD) diagram.
    """
    k = len(avranks)
    fig, ax = plt.subplots(figsize=(width, 5))

    # Set the range for ranks
    low, high = min(avranks) - 0.5, max(avranks) + 0.5
    ax.set_xlim(low, high)
    ax.set_ylim(-1, k + 1)

    # Plot the ranks
    for i, (rank, name) in enumerate(zip(avranks, names)):
        ax.scatter(rank, i, color='black', zorder=3)  # Add scatter point
        ax.text(rank, i, f'{name}', verticalalignment='center', horizontalalignment='right', fontsize=10)

    # Draw the CD bar
    cd_low, cd_high = low + (high - low) / 2 - cd / 2, low + (high - low) / 2 + cd / 2
    ax.hlines(-0.5, cd_low, cd_high, color='red', linewidth=2)
    ax.text((cd_low + cd_high) / 2, -1, f'CD = {cd:.2f}', color='red', horizontalalignment='center', fontsize=12)

    ax.axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()


# Wilcoxon-Holm 检验
def wilcoxon_holm(alpha=0.05, df_perf=None):
    """
    Performs Wilcoxon-Holm test with aligned data.
    """
    # 对数据进行对齐，只保留出现在所有分类器中的数据集
    df_perf = df_perf.pivot(index='Dataset', columns='Model', values='AUC')
    df_perf = df_perf.dropna()  # 删除含有 NaN 的行
    classifiers = df_perf.columns.tolist()

    # Friedman 检验
    friedman_p_value = friedmanchisquare(*[
        df_perf[c].values for c in classifiers
    ])[1]
    if friedman_p_value >= alpha:
        print('The null hypothesis over all classifiers cannot be rejected.')
        exit()

    # Wilcoxon 检验
    m = len(classifiers)
    p_values = []
    for i in range(m - 1):
        classifier_1 = classifiers[i]
        perf_1 = df_perf[classifier_1].values
        for j in range(i + 1, m):
            classifier_2 = classifiers[j]
            perf_2 = df_perf[classifier_2].values
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            p_values.append((classifier_1, classifier_2, p_value, False))

    # Holm 校正
    k = len(p_values)
    p_values.sort(key=lambda x: x[2])  # 按 p 值排序
    for i in range(k):
        new_alpha = alpha / (k - i)
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            break

    # 计算平均排名
    avg_ranks = df_perf.rank(axis=1, ascending=False).mean(axis=0).sort_values()
    return p_values, avg_ranks


# 主程序
if __name__ == "__main__":
    # 读取输入数据
    input_file = "result_add_std_deviation_evidence.xlsx"  # 输入文件
    df_perf = pd.read_excel(input_file)

    # 解析 AUC 并删除误差部分
    df_perf['AUC'] = df_perf['AUC'].apply(lambda x: float(str(x).split('±')[0]))

    # Wilcoxon-Holm 检验
    alpha = 0.05
    p_values, avg_ranks = wilcoxon_holm(alpha=alpha, df_perf=df_perf)

    # 设置 Critical Difference
    n_datasets = df_perf['Dataset'].nunique()
    k = len(avg_ranks)  # 分类器数量
    cd = 2.241 * np.sqrt((k * (k + 1)) / (6 * n_datasets))  # 使用 Tukey's HSD 方法计算 CD

    # 绘制 CD 图
    compute_critical_difference(avg_ranks.values, avg_ranks.index, cd)
