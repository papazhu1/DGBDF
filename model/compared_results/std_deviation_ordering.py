import pandas as pd
import numpy as np
import re

# 提取标准差的函数
def extract_std(value):
    try:
        # 使用正则表达式提取 "±" 后的数字
        std = re.search(r"±\s*([\d\.]+)", value)
        return float(std.group(1)) if std else np.nan
    except:
        return np.nan

# 计算稳定性排名
def calculate_stability_ranking(input_file, output_file):
    # 读取 Excel 文件
    df = pd.read_excel(input_file)

    # 提取标准差部分
    df['F1-std'] = df['F1-macro'].apply(extract_std)
    df['AUC-std'] = df['AUC'].apply(extract_std)
    df['AUPR-std'] = df['AUPR'].apply(extract_std)

    # 初始化存储排名的列表
    ranking_results = []

    # 遍历指标
    for metric_std in ['F1-std', 'AUC-std', 'AUPR-std']:
        all_ranks = []

        # 按数据集分组
        for dataset, group in df.groupby('Dataset'):
            group = group.copy()
            # 根据标准差排序，标准差越小排名越高
            group['Rank'] = group[metric_std].rank(ascending=True, method='min')
            all_ranks.append(group[['Model', 'Rank']])

        # 汇总所有排名，计算每个模型的平均排名
        combined_ranks = pd.concat(all_ranks)
        avg_ranking = combined_ranks.groupby('Model')['Rank'].mean().reset_index()
        avg_ranking = avg_ranking.rename(columns={'Rank': f'AvgRank_{metric_std}'})
        ranking_results.append(avg_ranking)

    # 合并所有指标的平均排名
    final_ranking = ranking_results[0]
    for i in range(1, len(ranking_results)):
        final_ranking = pd.merge(final_ranking, ranking_results[i], on='Model')

    # 计算最终排名：对平均排名再次排序
    final_ranking['Final_AvgRank'] = final_ranking.iloc[:, 1:].mean(axis=1)
    final_ranking = final_ranking.sort_values(by='Final_AvgRank', ascending=True).reset_index(drop=True)

    # 添加最终排名列
    final_ranking['Final_Rank'] = final_ranking.index + 1

    # 保存结果到 Excel 文件
    final_ranking.to_excel(output_file, index=False, sheet_name="Stability_Ranking")
    print(f"Stability rankings saved to {output_file}")

# 执行函数
if __name__ == "__main__":
    input_file = "result_add_std_deviation_evidence.xlsx"  # 输入文件
    output_file = "stability_ranking_final_evidence.xlsx"  # 输出文件
    calculate_stability_ranking(input_file, output_file)
