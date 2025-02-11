import pandas as pd
import re


def parse_tex_table(tex_file):
    with open(tex_file, 'r') as file:
        content = file.read()

    # 正则表达式匹配表格内容
    table_pattern = r"\\begin{tabular}.*?\\end{tabular}"
    table_content = re.search(table_pattern, content, re.DOTALL)

    if table_content:
        table = table_content.group(0)

        # 去掉多余的控制命令
        table = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", table)  # 去除像\textbf{}、\cellcolor{}这样的命令
        table = re.sub(r"\s+", " ", table)  # 去除多余空格

        # 分割每一行
        rows = table.split("\\\\")

        data = []
        for row in rows:
            # 分割每一列
            cols = row.split("&")
            # 去掉每个列值两侧的空格
            cols = [col.strip() for col in cols]
            data.append(cols)

        # 将数据转为 pandas DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])  # 假设第一行是列名
        return df
    else:
        print("未找到表格")
        return None


def save_to_excel(df, output_file):
    # 使用 pandas 写入 excel
    df.to_excel(output_file, index=False, engine='openpyxl')


# 读取 .tex 文件并解析表格
tex_file = r'C:\Users\10928\Documents\GitHub\DGBDF\model\compared_results\f1-macro_performance.tex'  # 请替换为你的文件路径
df = parse_tex_table(tex_file)

# 如果成功解析了表格，保存为 .xlsx 文件
if df is not None:
    save_to_excel(df, 'f1-macro_xlsx.xlsx')
    print("表格已成功保存为 aupr_xlsx.xlsx")
