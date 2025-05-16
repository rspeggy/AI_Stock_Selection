import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取汇总文件
summary_path = 'outputs/summary_metrics.csv'
summary_df = pd.read_csv(summary_path)

# 设置绘图风格
sns.set(style="whitegrid")

# 绘制 Accuracy, Precision, Recall, F1 的条形图
metrics = ['accuracy', 'precision', 'recall', 'f1']
for metric in metrics:
    plt.figure(figsize=(8, 4))
    sns.barplot(x='Ticker', y=metric, data=summary_df, palette='viridis')
    plt.title(f'{metric.capitalize()} by Stock')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Ticker')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'outputs/{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 绘制平均上涨概率线图
plt.figure(figsize=(8, 4))
sns.lineplot(x='Ticker', y='avg_up_prob', data=summary_df, marker='o', linestyle='-')
plt.title('Average Up Probability by Stock')
plt.ylabel('Average Up Probability')
plt.xlabel('Ticker')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/up_prob.png', dpi=300, bbox_inches='tight')
plt.close()
