import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# 提取标准差的函数
def extract_std(value):
    try:
        std = re.search(r"±\s*([\d\.]+)", value)
        return float(std.group(1)) if std else np.nan
    except:
        return np.nan


# 计算 Critical Difference (CD)
def compute_cd(num_models, num_datasets, alpha=0.05):
    q_alpha = norm.ppf(1 - alpha / 2)  # 双尾显著性 95% 的 z-score
    cd = q_alpha * np.sqrt(num_models * (num_models + 1) / (6 * num_datasets))
    return cd


# 绘制改进版的 Critical Difference Diagram
def plot_critical_difference_diagram_with_branches(models, average_ranks, cd, title="Critical Difference Diagram"):
    sorted_indices = np.argsort(average_ranks)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_ranks = [average_ranks[i] for i in sorted_indices]

    # 过滤掉 NaN 值
    sorted_ranks = [rank for rank in sorted_ranks if not np.isnan(rank)]
    sorted_models = [sorted_models[i] for i in range(len(sorted_ranks))]

    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=14)
    plt.xlabel("Average Ranks", fontsize=12)

    # 绘制横轴
    plt.hlines(1, min(sorted_ranks) - 1, max(sorted_ranks) + 1, colors='black', linewidth=1)
    plt.xticks(range(int(min(sorted_ranks) - 1), int(max(sorted_ranks) + 2)), fontsize=10)
    plt.yticks([])

    # 绘制分支连接线和模型名称
    for i, (rank, model) in enumerate(zip(sorted_ranks, sorted_models)):
        plt.scatter(rank, 1, s=100, color='blue')  # 标记点
        plt.text(rank, 1.05, model, fontsize=10, ha='center')  # 模型名称

        # 分支连接线
        if i > 0 and (rank - sorted_ranks[i - 1]) <= cd:
            plt.plot([sorted_ranks[i - 1], rank], [1, 1], color='black', linewidth=1)

    # 添加 CD 标记线
    plt.plot([sorted_ranks[0], sorted_ranks[0] + cd], [1.4, 1.4], color='red', lw=2)
    plt.text(sorted_ranks[0] + cd / 2, 1.5, f"CD = {cd:.2f}", color='red', fontsize=12, ha='center')

    plt.tight_layout()
    plt.show()


# 计算 AUC-std 稳定性排名
def calculate_auc_stability_ranking(input_file, output_file):
    df = pd.read_excel(input_file)
    df['AUC-std'] = df['AUC'].apply(extract_std)

    all_ranks = []

    for dataset, group in df.groupby('Dataset'):
        group = group.copy()
        group['Rank'] = group['AUC-std'].rank(ascending=True, method='min')
        all_ranks.append(group[['Model', 'Rank', 'AUC-std']])

    combined_ranks = pd.concat(all_ranks)

    # 过滤掉 NaN 值，避免计算错误
    combined_ranks = combined_ranks.dropna()

    avg_ranking = combined_ranks.groupby('Model')['Rank'].mean().reset_index()
    avg_ranking = avg_ranking.rename(columns={'Rank': 'AvgRank_AUC-std'})

    avg_ranking = avg_ranking.sort_values(by='AvgRank_AUC-std', ascending=True).reset_index(drop=True)
    avg_ranking['Final_Rank'] = avg_ranking.index + 1

    avg_ranking.to_excel(output_file, index=False, sheet_name="AUC_Stability_Ranking")
    print(f"AUC Stability rankings saved to {output_file}")

    return avg_ranking


if __name__ == "__main__":
    input_file = "result_add_std_deviation_evidence.xlsx"  # 输入文件
    output_file = "auc_std_deviation_wilcoxon.xlsx"  # 输出文件

    final_ranking = calculate_auc_stability_ranking(input_file, output_file)

    models = final_ranking['Model'].tolist()
    average_ranks = final_ranking['AvgRank_AUC-std'].tolist()
    num_datasets = 10  # 数据集数量

    # 计算 CD 值
    cd = compute_cd(len(models), num_datasets)

    # 绘制 CD 图
    plot_critical_difference_diagram_with_branches(models, average_ranks, cd, title="Critical Difference Diagram for AUC")
