import re

# LaTeX 文件路径
file_path = r"C:\Users\10928\Documents\GitHub\DGBDF\model\compared_results\gmean_performance.tex"

# 读取文件内容
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 处理每一行数据
processed_lines = []
for line in lines:
    # 只处理包含 AUC 结果的行（跳过表头等内容）
    if re.search(r" & ([\d\.]+) \$(\\pm)\$ ([\d\.]+)", line):
        # 提取所有 "AUC ± 标准差" 格式的数值
        values = re.findall(r"([\d\.]+) \$(\\pm)\$ ([\d\.]+)", line)
        auc_values = [float(v[0]) for v in values]  # 只提取 AUC 值（忽略标准差）

        # 找到该行的最大 AUC 值
        max_auc = max(auc_values)

        # 替换最大值，加上 `\cellcolor{graybg}\textbf{}`
        def highlight(match):
            auc, pm, std = match.groups()
            return f"\\cellcolor{{graybg}}\\textbf{{{auc} ${pm}$ {std}}}" if float(auc) == max_auc else match.group(0)

        line = re.sub(r"([\d\.]+) \$(\\pm)\$ ([\d\.]+)", highlight, line)

    processed_lines.append(line)

# 生成新的 LaTeX 文件
new_file_path = file_path.replace(".tex", "_modified.tex")
with open(new_file_path, "w", encoding="utf-8") as f:
    f.writelines(processed_lines)

print(f"修改完成，已保存至 {new_file_path}")
