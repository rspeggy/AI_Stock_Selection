import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Results folder path
results_folder = 'outputs'
summary_list = []

# Iterate through all CSV files
for filename in os.listdir(results_folder):
    if filename.startswith('predictions_') and filename.endswith('.csv'):
        stock_code = filename.split('_')[1].replace('.csv', '')
        file_path = os.path.join(results_folder, filename)
        df = pd.read_csv(file_path)
        avg_up_prob = df['Prob_Up'].mean()
        summary_list.append({'stock_code': stock_code, 'avg_up_prob': avg_up_prob})

# Create summary table
summary_df = pd.DataFrame(summary_list)
summary_df.sort_values(by='avg_up_prob', ascending=False, inplace=True)
summary_csv_path = os.path.join(results_folder, 'summary_table.csv')
summary_df.to_csv(summary_csv_path, index=False)
print(f"Summary table saved to: {summary_csv_path}")

# Select stocks with average up probability > 0.6
selected_stocks = summary_df[summary_df['avg_up_prob'] > 0.6]

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Set font for Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Create figure
plt.figure(figsize=(8, 5))
sns.set_style("whitegrid")  # Use seaborn's whitegrid style

# Plot bar chart
bars = plt.bar(selected_stocks['stock_code'], 
               selected_stocks['avg_up_prob'],
               color='#2E86C1',  # Set bar color
               alpha=0.8)  # Set transparency

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}',  # Display percentage
             ha='center', va='bottom')

# Set title and labels
plt.title('Stock Up Probability Analysis (>60%)', fontsize=16, pad=20)
plt.xlabel('Stock Code', fontsize=12, labelpad=10)
plt.ylabel('Average Up Probability', fontsize=12, labelpad=10)

# Set y-axis range and format
plt.ylim(0, 1)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# Set grid lines
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Adjust layout
plt.tight_layout()

# Save image (use higher DPI for clearer image)
plot_path = os.path.join(results_folder, 'selected_stocks_plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Visualization chart saved to: {plot_path}")

# Display summary table
print("\nStock Summary Table:")
print(summary_df.to_string())
