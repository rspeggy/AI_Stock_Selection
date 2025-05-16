import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 导入自定义因子
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_factors import HealthcareFactors, load_and_prepare_data

# === 参数 ===
LABEL_THRESHOLD = 0.05  # 5% 涨跌阈值
INVEST_THRESHOLD = 0.6  # 投资阈值：平均涨概率>60%才推荐

def get_stock_codes():
    raw_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
    stock_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    stock_codes = [os.path.splitext(f)[0] for f in stock_files]
    return stock_codes

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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
    )
    volatility = features.sector_volatility(
        stock_data['High'], stock_data['Low'], stock_data['Close']
    )
    momentum = features.momentum_score(
        stock_data['Close'], stock_data['Volume']
    )
    sentiment = features.healthcare_sentiment(
        stock_data['Close'], stock_data['Volume']
    )

    feature_df = pd.DataFrame({
        'volatility': volatility,
        'momentum': momentum,
        'sentiment': sentiment,
        'rsi': tech_indicators['rsi'],
        'macd': tech_indicators['macd'],
        'macd_signal': tech_indicators['macd_signal'],
        'bollinger_upper': tech_indicators['bollinger_upper'],
        'bollinger_lower': tech_indicators['bollinger_lower']
    })

    # === 构造三分类标签 ===
    future_return = stock_data['Close'].shift(-21) / stock_data['Close'] - 1  # 约一个月21个交易日
    label = future_return.apply(
        lambda x: 1 if x > LABEL_THRESHOLD else (-1 if x < -LABEL_THRESHOLD else 0)
    )

    # 划分训练/验证集
    train_start = config['train']['fit_start_time']
    train_end = config['train']['fit_end_time']
    valid_start = config['train']['valid_start_time']
    valid_end = config['train']['valid_end_time']

    train_data = feature_df.loc[train_start:train_end]
    train_label = label.loc[train_start:train_end]
    valid_data = feature_df.loc[valid_start:valid_end]
    valid_label = label.loc[valid_start:valid_end]

    train_mask = ~(train_data.isna().any(axis=1) | train_label.isna())
    valid_mask = ~(valid_data.isna().any(axis=1) | valid_label.isna())

    train_data = train_data[train_mask]
    train_label = train_label[train_mask]
    valid_data = valid_data[valid_mask]
    valid_label = valid_label[valid_mask]

    # === 修正标签 ===
    train_label = train_label + 1
    valid_label = valid_label + 1

    return train_data, train_label, valid_data, valid_label

def train_model(train_data, train_label, config):
    model = xgb.XGBClassifier(**config['model']['kwargs'])
    model.fit(train_data, train_label)
    return model

def evaluate_and_save_predictions(stock_code, model, valid_data, valid_label):
    pred = model.predict(valid_data)
    proba = model.predict_proba(valid_data)  # 返回三类概率

    results_df = valid_data.copy()
    results_df['true_label'] = valid_label.values
    results_df['pred_label'] = pred
    results_df['proba_down'] = proba[:, 0]
    results_df['proba_stable'] = proba[:, 1]
    results_df['proba_up'] = proba[:, 2]

    # 保存预测结果
    save_dir = "models/predictions"
    os.makedirs(save_dir, exist_ok=True)
    results_df.to_csv(os.path.join(save_dir, f"predictions_{stock_code}.csv"))

    # 汇总平均上涨概率
    avg_up = results_df['proba_up'].mean()

    metrics = {
        'accuracy': accuracy_score(valid_label, pred),
        'precision': precision_score(valid_label, pred, average='macro'),
        'recall': recall_score(valid_label, pred, average='macro'),
        'f1': f1_score(valid_label, pred, average='macro'),
        'avg_up_probability': avg_up
    }

    return metrics

def main():
    config = load_config("config/xgboost_config.yaml")
    stock_codes = get_stock_codes()
    print(f"找到 {len(stock_codes)} 只股票")

    invest_candidates = []

    for stock_code in stock_codes:
        print(f"\n处理股票: {stock_code}")
        train_data, train_label, valid_data, valid_label = prepare_data_for_stock(stock_code, config)

        if train_data is None or len(train_data) == 0:
            print(f"股票 {stock_code} 数据不足，跳过")
            continue

        print("正在训练模型...")
        model = train_model(train_data, train_label, config)

        print("正在评估和保存预测结果...")
        metrics = evaluate_and_save_predictions(stock_code, model, valid_data, valid_label)
        print(f"平均上涨概率: {metrics['avg_up_probability']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")

        if metrics['avg_up_probability'] > INVEST_THRESHOLD:
            invest_candidates.append(stock_code)

    print("\n⭐ 值得投资的股票（平均上涨概率 > {:.0%}）:".format(INVEST_THRESHOLD))
    for code in invest_candidates:
        print(f"- {code}")

if __name__ == "__main__":
    main()
