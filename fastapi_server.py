import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import qlib
from qlib.config import REG_CN
from qlib.data import D

# 导入自定义因子
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.custom_factors import HealthcareFactors, create_custom_factors

# 导入模型
from models.train_xgb import load_model as load_xgb_model
from models.train_logistic import load_model as load_logistic_model
from optimization.optimizer import optimize_healthcare_portfolio

app = FastAPI(
    title="医疗健康股票分析API",
    description="基于Qlib的医疗健康股票分析系统API",
    version="1.0.0"
)

# 加载配置
with open("config/workflow_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# 初始化Qlib
qlib.init(provider_uri=config['data']['provider_uri'], region=REG_CN)

# 加载模型
xgb_model = load_xgb_model("models/saved/xgb_model_latest.pkl")
logistic_model = load_logistic_model("models/saved/logistic_model_latest.pkl")

class StockRequest(BaseModel):
    """股票请求模型"""
    symbols: List[str]
    start_date: str
    end_date: str

class PredictionResponse(BaseModel):
    """预测响应模型"""
    predictions: Dict[str, float]
    probabilities: Dict[str, float]
    timestamp: str

class PortfolioRequest(BaseModel):
    """投资组合请求模型"""
    symbols: List[str]
    start_date: str
    end_date: str
    risk_free_rate: Optional[float] = 0.02
    max_weight: Optional[float] = 0.3
    min_weight: Optional[float] = 0.0

class PortfolioResponse(BaseModel):
    """投资组合响应模型"""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    timestamp: str

@app.get("/")
def read_root():
    """API根路径"""
    return {"message": "欢迎使用医疗健康股票分析API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: StockRequest):
    """
    预测股票涨跌
    
    Args:
        request (StockRequest): 请求参数
        
    Returns:
        PredictionResponse: 预测结果
    """
    try:
        # 获取数据
        data = D.features(
            request.symbols,
            [f['expression'] for f in config['features']],
            start_time=request.start_date,
            end_time=request.end_date
        )
        
        # 创建自定义因子
        factors = create_custom_factors(data)
        
        # 预测
        xgb_pred = xgb_model.predict(factors)
        logistic_pred = logistic_model.predict(factors)
        
        # 计算概率
        xgb_prob = xgb_model.predict_proba(factors)
        logistic_prob = logistic_model.predict_proba(factors)
        
        # 组合预测结果
        predictions = {}
        probabilities = {}
        for symbol in request.symbols:
            # 取两个模型的平均预测
            pred = (xgb_pred[symbol].iloc[-1] + logistic_pred[symbol].iloc[-1]) / 2
            prob = (xgb_prob[symbol].iloc[-1] + logistic_prob[symbol].iloc[-1]) / 2
            predictions[symbol] = float(pred)
            probabilities[symbol] = float(prob)
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize_portfolio", response_model=PortfolioResponse)
def optimize_portfolio(request: PortfolioRequest):
    """
    优化投资组合
    
    Args:
        request (PortfolioRequest): 请求参数
        
    Returns:
        PortfolioResponse: 优化结果
    """
    try:
        # 获取数据
        data = D.features(
            request.symbols,
            ["$close"],
            start_time=request.start_date,
            end_time=request.end_date
        )
        
        # 计算收益率
        returns = data.pct_change().dropna()
        
        # 优化投资组合
        result = optimize_healthcare_portfolio(
            returns,
            risk_free_rate=request.risk_free_rate,
            max_weight=request.max_weight,
            min_weight=request.min_weight
        )
        
        return PortfolioResponse(
            weights=result['optimal_portfolio']['weights'],
            expected_return=float(result['optimal_portfolio']['expected_return']),
            volatility=float(result['optimal_portfolio']['volatility']),
            sharpe_ratio=float(result['optimal_portfolio']['sharpe_ratio']),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_server:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api']['debug']
    ) 