import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 导入自定义因子
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_factors import HealthcareFactors, load_and_prepare_data, process_all_stocks

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_stock_codes():
    raw_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
    stock_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    stock_codes = [os.path.splitext(f)[0] for f in stock_files]
    return stock_codes

def load_sentiment_data(stock_code):
    sentiment_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'edgar', 'edgar_sentiment_scores.csv')
    if not os.path.exists(sentiment_file):
        return None
    sentiment_data = pd.read_csv(sentiment_file)
    sentiment_data = sentiment_data[sentiment_data['ticker'] == stock_code]
    if len(sentiment_data) == 0:
        return None
    sentiment_data['sentiment_score'] = (
        sentiment_data['positive'] - 
        sentiment_data['negative'] + 
        sentiment_data['uncertainty'] * 0.5 - 
        sentiment_data['litigious'] * 0.5
    )
    sentiment_by_type = sentiment_data.groupby('report_type')['sentiment_score'].mean()
    sentiment_factors = pd.DataFrame({
        'sentiment_10k': sentiment_by_type.get('10-K', 0),
        'sentiment_10q': sentiment_by_type.get('10-Q', 0)
    }, index=[sentiment_data.index[0]])
    return sentiment_factors

def prepare_data_for_stock(stock_code, config):
    stock_data, macro_data = load_and_prepare_data(
        stock_code,
        start_date=config['data']['start_time'],
        end_date=config['data']['end_time']
    )
    if stock_data is None:
        return None, None, None, None

    features = HealthcareFactors()
    tech_indicators = features.technical_indicators(
        stock_data['High'],
        stock_data['Low'],
        stock_data['Close'],
        stock_data['Volume']
    ).fillna(method='bfill')  # ✅ 新增：向后填充NaN

    volatility = features.sector_volatility(
        stock_data['High'],
        stock_data['Low'],
        stock_data['Close']
    )
    momentum = features.momentum_score(
        stock_data['Close'],
        stock_data['Volume']
    )
    sentiment = features.healthcare_sentiment(
        stock_data['Close'],
        stock_data['Volume']
    )
    sentiment_factors = load_sentiment_data(stock_code)

    if macro_data is not None:
        macro_corr = features.macro_impact_factor(stock_data, macro_data)
        macro_corr = pd.DataFrame({'macro_impact': macro_corr})
    else:
        macro_corr = pd.DataFrame({'macro_impact': 0}, index=stock_data.index)

    feature_df = pd.DataFrame({
        'volatility': volatility,
        'momentum': momentum,
        'sentiment': sentiment,
        'rsi': tech_indicators['rsi'],
        'macd': tech_indicators['macd'],
        'macd_signal': tech_indicators['macd_signal'],
        'bollinger_upper': tech_indicators['bollinger_upper'],
        'bollinger_lower': tech_indicators['bollinger_lower']
    }).join(macro_corr)

    if sentiment_factors is not None:
        feature_df['sentiment_10k'] = sentiment_factors['sentiment_10k'].iloc[0]
        feature_df['sentiment_10q'] = sentiment_factors['sentiment_10q'].iloc[0]

    label = stock_data['Close'].shift(-1) / stock_data['Close'] - 1
    label = (label > 0).astype(int)

    train_start = pd.to_datetime(config['train']['fit_start_time'])
    train_end = pd.to_datetime(config['train']['fit_end_time'])
    valid_start = pd.to_datetime(config['train']['valid_start_time'])
    valid_end = pd.to_datetime(config['train']['valid_end_time'])

    print(f"\n日期范围信息:")
    print(f"训练集: {train_start} 到 {train_end}")
    print(f"验证集: {valid_start} 到 {valid_end}")
    print(f"特征数据索引范围: {feature_df.index.min()} 到 {feature_df.index.max()}")  # ✅ 新增调试
    print(f"数据范围: {feature_df.index.min()} 到 {feature_df.index.max()}")

    train_data = feature_df.loc[train_start:train_end]
    train_label = label.loc[train_start:train_end]
    valid_data = feature_df.loc[valid_start:valid_end]
    valid_label = label.loc[valid_start:valid_end]

    print(f"\n数据形状信息:")
    print(f"训练数据: {train_data.shape}")
    print(f"训练标签: {train_label.shape}")
    print(f"验证数据: {valid_data.shape}")
    print(f"验证标签: {valid_label.shape}")

    # ✅ 改进：仅关键列严格去除NaN
    critical_columns = ['volatility', 'momentum', 'sentiment', 'macro_impact']
    train_mask = ~(train_data[critical_columns].isna().any(axis=1) | train_label.isna())
    valid_mask = ~(valid_data[critical_columns].isna().any(axis=1) | valid_label.isna())

    train_data = train_data[train_mask]
    train_label = train_label[train_mask]
    valid_data = valid_data[valid_mask]
    valid_label = valid_label[valid_mask]

    print(f"\n过滤后的数据形状:")
    print(f"训练数据: {train_data.shape}")
    print(f"训练标签: {train_label.shape}")
    print(f"验证数据: {valid_data.shape}")
    print(f"验证标签: {valid_label.shape}")

    if len(train_data) == 0 or len(valid_data) == 0:
        print(f"\n警告: 股票 {stock_code} 的训练或验证数据为空")
        print("训练数据日期范围:", train_start, "到", train_end)
        print("验证数据日期范围:", valid_start, "到", valid_end)
        print("原始特征数据日期范围:", feature_df.index.min(), "到", feature_df.index.max())
        return None, None, None, None

    min_samples = 30
    if len(train_data) < min_samples or len(valid_data) < min_samples:
        print(f"\n警告: 股票 {stock_code} 的训练或验证数据样本数不足")
        print(f"训练数据样本数: {len(train_data)}")
        print(f"验证数据样本数: {len(valid_data)}")
        return None, None, None, None

    scaler = StandardScaler()
    train_data_scaled = pd.DataFrame(scaler.fit_transform(train_data), index=train_data.index, columns=train_data.columns)
    valid_data_scaled = pd.DataFrame(scaler.transform(valid_data), index=valid_data.index, columns=valid_data.columns)

    return train_data_scaled, train_label, valid_data_scaled, valid_label

def train_model(train_data, train_label, config):
    model = LogisticRegression(**config['model']['kwargs'])
    model.fit(train_data, train_label)
    return model

def evaluate_model(model, valid_data, valid_label):
    pred = model.predict(valid_data)
    metrics = {
        'accuracy': accuracy_score(valid_label, pred),
        'precision': precision_score(valid_label, pred),
        'recall': recall_score(valid_label, pred),
        'f1': f1_score(valid_label, pred)
    }
    return metrics

def save_results(stock_code, model, metrics, config):
    save_dir = "models/saved"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"logistic_model_{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    pd.to_pickle(model, model_path)
    results = {
        'stock_code': stock_code,
        'config': config,
        'metrics': metrics
    }
    results_path = os.path.join(save_dir, f"results_{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    print(f"模型和结果已保存到 {save_dir}")

def main():
    config = load_config("config/logistic_config.yaml")
    stock_codes = get_stock_codes()
    print(f"找到 {len(stock_codes)} 只股票")
    for stock_code in stock_codes:
        print(f"\n处理股票: {stock_code}")
        print("正在准备数据...")
        train_data, train_label, valid_data, valid_label = prepare_data_for_stock(stock_code, config)
        if train_data is None or len(train_data) == 0:
            print(f"股票 {stock_code} 数据不足，跳过")
            continue
        print("正在训练模型...")
        model = train_model(train_data, train_label, config)
        print("正在评估模型...")
        metrics = evaluate_model(model, valid_data, valid_label)
        print("\n评估指标:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        save_results(stock_code, model, metrics, config)

if __name__ == "__main__":
    main()
