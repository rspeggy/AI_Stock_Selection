import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义因子
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_factors import HealthcareFactors, load_and_prepare_data

def load_model(model_path):
    """
    加载模型
    
    Args:
        model_path (str): 模型文件路径
        
    Returns:
        模型对象
    """
    return pd.read_pickle(model_path)

def prepare_prediction_data(stock_code, start_date, end_date):
    """
    准备预测数据
    
    Args:
        stock_code (str): 股票代码
        start_date (str): 开始日期
        end_date (str): 结束日期
        
    Returns:
        pd.DataFrame: 预测数据
    """
    # 加载数据
    stock_data, macro_data = load_and_prepare_data(
        stock_code,
        start_date=start_date,
        end_date=end_date
    )
    
    if stock_data is None:
        return None
    
    # 创建特征
    features = HealthcareFactors()
    
    # 计算技术指标
    tech_indicators = features.technical_indicators(
        stock_data['High'],
        stock_data['Low'],
        stock_data['Close'],
        stock_data['Volume']
    )
    
    # 计算其他因子
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
    
    # 合并所有特征
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
    
    # 删除包含NaN的行
    feature_df = feature_df.dropna()
    
    return feature_df

def predict(model, data):
    """
    使用模型进行预测
    
    Args:
        model: 训练好的模型
        data (pd.DataFrame): 预测数据
        
    Returns:
        pd.Series: 预测结果（1表示上涨，0表示下跌）
    """
    return pd.Series(model.predict(data), index=data.index)

def save_predictions(stock_code, predictions, model_type):
    """
    保存预测结果
    
    Args:
        stock_code (str): 股票代码
        predictions (pd.Series): 预测结果
        model_type (str): 模型类型（'logistic' 或 'xgb'）
    """
    # 创建保存目录
    save_dir = "predictions"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存预测结果
    predictions_df = pd.DataFrame({
        'date': predictions.index,
        'prediction': predictions.values
    })
    
    predictions_path = os.path.join(
        save_dir, 
        f"{model_type}_predictions_{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    predictions_df.to_csv(predictions_path, index=False)
    print(f"预测结果已保存到 {predictions_path}")

def main():
    # 设置预测参数
    stock_code = "AAPL"  # 示例股票代码
    start_date = "2024-01-01"
    end_date = "2024-03-21"
    model_type = "xgb"  # 或 "logistic"
    
    # 加载模型
    model_dir = "models/saved"
    model_files = [f for f in os.listdir(model_dir) if f.startswith(f"{model_type}_model_{stock_code}")]
    if not model_files:
        print(f"未找到股票 {stock_code} 的{model_type}模型")
        return
    
    # 使用最新的模型
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    model = load_model(model_path)
    print(f"已加载模型: {latest_model}")
    
    # 准备预测数据
    print("正在准备预测数据...")
    prediction_data = prepare_prediction_data(stock_code, start_date, end_date)
    if prediction_data is None or len(prediction_data) == 0:
        print("数据不足，无法进行预测")
        return
    
    # 进行预测
    print("正在进行预测...")
    predictions = predict(model, prediction_data)
    
    # 打印预测结果
    print("\n预测结果:")
    print(f"预测上涨天数: {predictions.sum()}")
    print(f"预测下跌天数: {len(predictions) - predictions.sum()}")
    print(f"上涨概率: {predictions.mean():.2%}")
    
    # 保存预测结果
    save_predictions(stock_code, predictions, model_type)

if __name__ == "__main__":
    main() 