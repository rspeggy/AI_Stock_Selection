import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Tuple
import yfinance as yf
from datetime import datetime, timedelta

class MarkowitzOptimizer:
    """Markowitz投资组合优化器"""
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        初始化优化器
        
        Args:
            returns (pd.DataFrame): 收益率数据
            risk_free_rate (float): 无风险利率
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)
    
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        计算投资组合表现
        
        Args:
            weights (np.ndarray): 资产权重
            
        Returns:
            Tuple[float, float, float]: (预期收益率, 波动率, 夏普比率)
        """
        returns = np.sum(self.mean_returns * weights)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (returns - self.risk_free_rate) / volatility
        return returns, volatility, sharpe_ratio
    
    def negative_sharpe(self, weights: np.ndarray) -> float:
        """
        计算负夏普比率（用于最小化）
        
        Args:
            weights (np.ndarray): 资产权重
            
        Returns:
            float: 负夏普比率
        """
        returns, volatility, _ = self.portfolio_performance(weights)
        return -(returns - self.risk_free_rate) / volatility
    
    def optimize_portfolio(self, 
                         target_return: float = None,
                         target_volatility: float = None,
                         max_weight: float = 0.3,
                         min_weight: float = 0.0) -> Dict:
        """
        优化投资组合
        
        Args:
            target_return (float): 目标收益率
            target_volatility (float): 目标波动率
            max_weight (float): 最大权重
            min_weight (float): 最小权重
            
        Returns:
            Dict: 优化结果
        """
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # 权重和为1
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self.portfolio_performance(x)[0] - target_return
            })
        
        if target_volatility is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self.portfolio_performance(x)[1] - target_volatility
            })
        
        # 边界条件
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # 初始权重
        init_weights = np.array([1/self.n_assets] * self.n_assets)
        
        # 优化
        result = minimize(
            self.negative_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # 计算优化后的投资组合表现
        returns, volatility, sharpe = self.portfolio_performance(result.x)
        
        return {
            'weights': dict(zip(self.returns.columns, result.x)),
            'expected_return': returns,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'success': result.success,
            'message': result.message
        }
    
    def efficient_frontier(self, 
                         n_points: int = 100,
                         max_weight: float = 0.3,
                         min_weight: float = 0.0) -> pd.DataFrame:
        """
        计算有效前沿
        
        Args:
            n_points (int): 点数
            max_weight (float): 最大权重
            min_weight (float): 最小权重
            
        Returns:
            pd.DataFrame: 有效前沿数据
        """
        # 计算最小和最大收益率
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        
        # 生成目标收益率序列
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        # 计算有效前沿
        efficient_portfolios = []
        for target_return in target_returns:
            try:
                result = self.optimize_portfolio(
                    target_return=target_return,
                    max_weight=max_weight,
                    min_weight=min_weight
                )
                if result['success']:
                    efficient_portfolios.append({
                        'target_return': target_return,
                        'expected_return': result['expected_return'],
                        'volatility': result['volatility'],
                        'sharpe_ratio': result['sharpe_ratio']
                    })
            except:
                continue
        
        return pd.DataFrame(efficient_portfolios)

def optimize_healthcare_portfolio(returns: pd.DataFrame,
                                risk_free_rate: float = 0.02,
                                max_weight: float = 0.3,
                                min_weight: float = 0.0) -> Dict:
    """
    优化医疗健康行业投资组合
    
    Args:
        returns (pd.DataFrame): 收益率数据
        risk_free_rate (float): 无风险利率
        max_weight (float): 最大权重
        min_weight (float): 最小权重
        
    Returns:
        Dict: 优化结果
    """
    optimizer = MarkowitzOptimizer(returns, risk_free_rate)
    
    # 优化投资组合
    result = optimizer.optimize_portfolio(
        max_weight=max_weight,
        min_weight=min_weight
    )
    
    # 计算有效前沿
    efficient_frontier = optimizer.efficient_frontier(
        n_points=100,
        max_weight=max_weight,
        min_weight=min_weight
    )
    
    return {
        'optimal_portfolio': result,
        'efficient_frontier': efficient_frontier
    }

if __name__ == "__main__":
 
    symbols = ["ENPH", "PYPL", "GH", "SNOW", "PLUG", "FSLR"]

    data_frames = []
    for symbol in symbols:
        # 读取第一行作为列名，并将第一个列名替换为'Date'
        columns = pd.read_csv(f"data/raw/{symbol}.csv", nrows=0).columns
        columns = ['Date'] + list(columns[1:])
        # 跳过前两行（Price和Ticker行）和第三行（空行）
        df = pd.read_csv(f"data/raw/{symbol}.csv", skiprows=3, names=columns)
        # 将Date列设置为索引
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        data_frames.append(df['Close'].rename(symbol))

    # 合并成一个DataFrame，只保留所有股票都存在的日期
    adj_close = pd.concat(data_frames, axis=1, join='inner')
    print("adj_close shape:", adj_close.shape)
    print("adj_close head:\n", adj_close.head())

    # 计算收益率
    returns = adj_close.pct_change().dropna()
    
    # 优化投资组合
    result = optimize_healthcare_portfolio(returns)
    
    # 打印结果
    print("\n最优投资组合:")
    print("权重:")
    for symbol, weight in result['optimal_portfolio']['weights'].items():
        print(f"{symbol}: {weight:.2%}")
    print(f"\n预期收益率: {result['optimal_portfolio']['expected_return']:.2%}")
    print(f"波动率: {result['optimal_portfolio']['volatility']:.2%}")
    print(f"夏普比率: {result['optimal_portfolio']['sharpe_ratio']:.2f}") 