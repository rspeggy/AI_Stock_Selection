import pandas as pd
import os
import glob

# 路径配置
score_csv = "/Users/jpeng/Downloads/qlib_ai_healthcare_project/data/edgar/edgar_sentiment_scores_v0.csv"
filings_dir = "/Users/jpeng/Downloads/qlib_ai_healthcare_project/data/edgar/sec-edgar-filings"

# 加载情感分数文件
df = pd.read_csv(score_csv)

# 收集所有 txt 文件路径（不排序，完全按 glob 顺序）
txt_files = glob.glob(os.path.join(filings_dir, "**", "*.txt"), recursive=True)

# 检查数量是否一致
if len(txt_files) != len(df):
    raise ValueError(f"文件数不匹配：txt_files={len(txt_files)} vs CSV行数={len(df)}")

# 准备新列
full_paths = []
year_list = []
publish_dates = []

for i, (idx, row) in enumerate(df.iterrows()):
    path = txt_files[i]
    full_paths.append(path)

    # 从倒数第二级目录提取年份和顺序号
    folder = path.split(os.sep)[-2]  # e.g., 0001564590-20-033670
    year = "20" + folder.split('-')[1]  # e.g., '-20-' → 2020
    year_list.append(int(year))

    report_type = row['report_type']

    # 推导发布日期
    if report_type == "10-K":
        publish_date = f"{year}-12-31"
    elif report_type == "10-Q":
        # 收集同 ticker、report_type、year 的所有路径（用 glob 顺序）
        ticker = row['ticker']
        same_group = []
        
        # 遍历所有文件，找到匹配的文件
        for p in txt_files:
            # 检查文件路径是否包含正确的 ticker 和 report_type
            if f"/{ticker}/{report_type}/" in p:
                # 从文件路径中提取年份
                p_folder = p.split(os.sep)[-2]
                p_year = "20" + p_folder.split('-')[1]
                # 如果年份匹配，添加到组中
                if p_year == year:
                    same_group.append(p)
        
        # 按文件路径排序，确保顺序一致
        same_group.sort()
        
        # 在这一组中找当前 path 的索引位置
        try:
            rank_in_year = same_group.index(path)
        except ValueError:
            print(f"Warning: Could not find path in same_group for {ticker} {report_type} {year}")
            print(f"Current path: {path}")
            print(f"Available paths: {same_group}")
            rank_in_year = 0  # 使用默认值

        if rank_in_year == 0:
            publish_date = f"{year}-03-31"
        elif rank_in_year == 1:
            publish_date = f"{year}-06-30"
        elif rank_in_year == 2:
            publish_date = f"{year}-09-30"
        else:
            print(f"Warning: more than 3 10-Q filings in {year} for {ticker}")
            publish_date = None
    else:
        publish_date = None

    publish_dates.append(publish_date)

# 添加新列到 DataFrame
df['full_path'] = full_paths
df['year'] = year_list
df['publish_date'] = publish_dates

# 保存新文件
output_csv = "data/edgar/edgar_sentiment_scores_with_paths_and_dates.csv"
df.to_csv(output_csv, index=False)
print(f"Saved updated file with paths and dates to {output_csv}")
