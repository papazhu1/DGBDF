import os
import numpy as np
import pandas as pd


def process_all_results_evidence(base_path):
    results = []  # 用于存储所有结果

    # 读取Excel文件
    file_path = os.path.join(base_path, "result_add_std_deviation_evidence.xlsx")
    df = pd.read_excel(file_path)

    # 获取包含 UncertaintyAwareDeepForest 的数据集
    valid_datasets = df[df['Model'] == 'UncertaintyAwareDeepForest']['Dataset'].unique()

    # 过滤数据集，只保留包含 UncertaintyAwareDeepForest 的数据集
    filtered_df = df[df['Dataset'].isin(valid_datasets)]

    # 保存到新的Excel文件
    new_file_path = os.path.join(base_path, "filtered_result_add_std_deviation_evidence.xlsx")
    filtered_df.to_excel(new_file_path, index=False)
    print(f"Filtered results saved to {new_file_path}")


def generate_latex_table(base_path, metric):
    # 读取过滤后的Excel文件
    file_path = os.path.join(base_path, "filtered_result_add_std_deviation_evidence.xlsx")
    df = pd.read_excel(file_path)

    # 只保留所需性能指标的数据
    df = df[['Dataset', 'Model', metric]]

    # 仅保留前10个模型
    top_models = df['Model'].unique()[[0, 3, 4, 1, 13, 14, 8, 9, 15]]
    df = df[df['Model'].isin(top_models)]
    df['Model'] = pd.Categorical(df['Model'], categories=top_models, ordered=True)
    df = df.sort_values('Model')

    # 处理数据集名称，添加 \_ 以兼容 LaTeX
    df['Dataset'] = df['Dataset'].str.replace('_', '\_', regex=True)

    # 获取模型名称
    model_names = df['Model'].unique().tolist()
    column_headers = ' & '.join(model_names)

    # 生成 LaTeX 表格代码，确保 `\bottomrule` 被正确解析
    latex_code = (
        "\\begin{table*}[htbp]\n"
        "    \\centering\n"
        f"    \\caption{{{metric} Performance Comparison}}\n"
        f"    \\label{{tab:{metric.lower()}}}\n"
        f"    \\begin{{tabular}}{{l{'c' * len(model_names)}}}\n"
        "        \\toprule\n"
        f"        Dataset & {column_headers} \\\\ \n"
        "        \\midrule\n"
    )

    # 重新格式化数据
    grouped_df = df.pivot(index='Dataset', columns='Model', values=metric).reset_index()

    for _, row in grouped_df.iterrows():
        dataset = row['Dataset']
        models = row.drop('Dataset').fillna('').astype(str).tolist()
        models = [m.replace("±", "$\\pm$") for m in models]  # 替换LaTeX不支持的符号
        latex_code += f"            {dataset} & " + " & ".join(models) + " \\\\ \n"

    latex_code += (
        "        \\bottomrule\n"
        "    \\end{tabular}\n"
        "\\end{table*}\n"
    )

    # 保存 LaTeX 代码到文件
    latex_file_path = os.path.join(base_path, f"{metric.lower()}_performance.tex")
    with open(latex_file_path, "w") as f:
        f.write(latex_code)
    print(f"LaTeX table for {metric} saved to {latex_file_path}")


if __name__ == "__main__":
    base_path = os.getcwd()  # 获取当前工作目录
    process_all_results_evidence(base_path)

    # 生成多个性能指标的 LaTeX 表格
    metrics = ["F1-macro", "AUC", "AUPR", "Gmean"]
    for metric in metrics:
        generate_latex_table(base_path, metric)
