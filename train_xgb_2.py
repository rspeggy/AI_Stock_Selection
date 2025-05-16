import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 导入自定义因子
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_factors import HealthcareFactors, load_and_prepare_data

def get_stock_codes():
    raw_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
    stock_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    return [os.path.splitext(f)[0] for f in stock_files]

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_data_for_stock(stock_code, config):
    stock_data, macro_data = load_and_prepare_data(
        stock_code,
        start_date=config['train']['start_time'],
        end_date=config['train']['end_time']
    )
    if stock_data is None:
        return None, None, None, None, None

    features = HealthcareFactors()
    tech_indicators = features.technical_indicators(
        stock_data['High'], stock_data['Low'], stock_data['Close'], stock_data['Volume'])
    volatility = features.sector_volatility(stock_data['High'], stock_data['Low'], stock_data['Close'])
    momentum = features.momentum_score(stock_data['Close'], stock_data['Volume'])
    sentiment = features.healthcare_sentiment(stock_data['Close'], stock_data['Volume'])

    feature_df = pd.DataFrame({
        'volatility': volatility,
        'momentum': momentum,
        'sentiment': sentiment,
        'rsi': tech_indicators['rsi'],
        'macd': tech_indicators['macd'],
        'macd_signal': tech_indicators['macd_signal'],
        'bollinger_upper': tech_indicators['bollinger_upper'],
        'bollinger_lower': tech_indicators['bollinger_lower']
    }, index=stock_data.index)

    label = stock_data['Close'].shift(-21) / stock_data['Close'] - 1
    threshold = 0.02
    label = label.apply(lambda x: 2 if x > threshold else (1 if x < -threshold else 0))

    train_data = feature_df.loc[config['train']['fit_start_time']:config['train']['fit_end_time']]
    train_label = label.loc[config['train']['fit_start_time']:config['train']['fit_end_time']]
    valid_data = feature_df.loc[config['train']['valid_start_time']:config['train']['valid_end_time']]
    valid_label = label.loc[config['train']['valid_start_time']:config['train']['valid_end_time']]

    mask_train = ~(train_data.isna().any(axis=1) | train_label.isna())
    mask_valid = ~(valid_data.isna().any(axis=1) | valid_label.isna())
    return (train_data[mask_train], train_label[mask_train], 
            valid_data[mask_valid], valid_label[mask_valid], 
            valid_data[mask_valid].index)

def train_model(train_data, train_label, config):
    model = xgb.XGBClassifier(**config['model']['kwargs'])
    model.fit(train_data, train_label)
    return model

def evaluate_and_save_predictions(stock_code, model, valid_data, valid_label, valid_dates):
    pred = model.predict(valid_data)
    pred_proba = model.predict_proba(valid_data)

    df = pd.DataFrame({
        'Date': valid_dates,
        'Ticker': stock_code,
        'True_Label': valid_label.values,
        'Predicted_Label': pred,
        'Prob_Up': pred_proba[:, 2],
        'Prob_Stable': pred_proba[:, 0],
        'Prob_Down': pred_proba[:, 1]
    })

    os.makedirs('outputs', exist_ok=True)
    csv_path = f'outputs/predictions_{stock_code}.csv'
    df.to_csv(csv_path, index=False)
    print(f"预测结果已保存到 {csv_path}")

    avg_up = df['Prob_Up'].mean()
    metrics = {
        'accuracy': accuracy_score(valid_label, pred),
        'precision': precision_score(valid_label, pred, average='weighted'),
        'recall': recall_score(valid_label, pred, average='weighted'),
        'f1': f1_score(valid_label, pred, average='weighted'),
        'avg_up_prob': avg_up
    }

    metrics_path = f'outputs/metrics_{stock_code}.csv'
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"评估指标已保存到 {metrics_path}")

    return avg_up, metrics

def main():
    config = load_config("xgboost_config.yaml")
    stock_codes = get_stock_codes()
    print(f"找到 {len(stock_codes)} 只股票")

    worthy_stocks = []
    all_metrics = []

    for stock_code in stock_codes:
        print(f"\n处理股票: {stock_code}")
        train_data, train_label, valid_data, valid_label, valid_dates = prepare_data_for_stock(stock_code, config)
        if train_data is None or len(train_data) == 0:
            print(f"股票 {stock_code} 数据不足，跳过")
            continue

        model = train_model(train_data, train_label, config)
        avg_up, metrics = evaluate_and_save_predictions(stock_code, model, valid_data, valid_label, valid_dates)
        all_metrics.append({'Ticker': stock_code, **metrics})

        if avg_up > 0.6:
            worthy_stocks.append(stock_code)

    summary_df = pd.DataFrame(all_metrics)
    summary_path = 'outputs/summary_metrics.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n所有股票汇总指标已保存到 {summary_path}")

    print("\n⭐ 值得投资的股票（平均上涨概率 > 60%）:")
    for stock in worthy_stocks:
        print(f"- {stock}")

if __name__ == "__main__":
    main()
