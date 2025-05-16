import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import datetime

def load_backtest_results(results_path: str) -> Dict:
    """
    加载回测结果
    
    Args:
        results_path (str): 结果文件路径
        
    Returns:
        Dict: 回测结果
    """
    with open(results_path, 'r') as f:
        results = yaml.safe_load(f)
    return results

def calculate_metrics(returns: pd.Series) -> Dict:
    """
    计算评估指标
    
    Args:
        returns (pd.Series): 收益率序列
        
    Returns:
        Dict: 评估指标
    """
    # 计算年化收益率
    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    
    # 计算年化波动率
    annual_volatility = returns.std() * np.sqrt(252)
    
    # 计算夏普比率
    risk_free_rate = 0.02  # 假设无风险利率为2%
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    # 计算最大回撤
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # 计算索提诺比率
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_volatility
    
    # 计算卡玛比率
    calmar_ratio = annual_return / abs(max_drawdown)
    
    # 计算胜率
    win_rate = len(returns[returns > 0]) / len(returns)
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate
    }

def plot_returns(returns: pd.Series, benchmark_returns: pd.Series = None, title: str = "策略收益"):
    """
    绘制收益率曲线
    
    Args:
        returns (pd.Series): 策略收益率
        benchmark_returns (pd.Series): 基准收益率
        title (str): 图表标题
    """
    plt.figure(figsize=(12, 6))
    
    # 计算累积收益率
    cum_returns = (1 + returns).cumprod()
    plt.plot(cum_returns.index, cum_returns.values, label='策略')
    
    if benchmark_returns is not None:
        cum_benchmark = (1 + benchmark_returns).cumprod()
        plt.plot(cum_benchmark.index, cum_benchmark.values, label='基准', linestyle='--')
    
    plt.title(title)
    plt.xlabel('日期')
    plt.ylabel('累积收益率')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_drawdown(returns: pd.Series, title: str = "回撤分析"):
    """
    绘制回撤曲线
    
    Args:
        returns (pd.Series): 收益率序列
        title (str): 图表标题
    """
    plt.figure(figsize=(12, 6))
    
    # 计算回撤
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    
    plt.plot(drawdowns.index, drawdowns.values)
    plt.title(title)
    plt.xlabel('日期')
    plt.ylabel('回撤')
    plt.grid(True)
    plt.show()

def plot_monthly_returns(returns: pd.Series, title: str = "月度收益热力图"):
    """
    绘制月度收益热力图
    
    Args:
        returns (pd.Series): 收益率序列
        title (str): 图表标题
    """
    # 计算月度收益率
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # 创建月度收益矩阵
    monthly_returns_matrix = monthly_returns.to_frame()
    monthly_returns_matrix.index = pd.MultiIndex.from_arrays([
        monthly_returns_matrix.index.year,
        monthly_returns_matrix.index.month
    ])
    monthly_returns_matrix = monthly_returns_matrix.unstack()
    
    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(monthly_returns_matrix, annot=True, fmt='.2%', cmap='RdYlGn', center=0)
    plt.title(title)
    plt.xlabel('月份')
    plt.ylabel('年份')
    plt.show()

def analyze_strategy(results: Dict, save_dir: str = "backtest/results"):
    """
    分析策略表现
    
    Args:
        results (Dict): 回测结果
        save_dir (str): 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取收益率
    returns = pd.Series(results['returns'])
    benchmark_returns = pd.Series(results['benchmark_returns'])
    
    # 计算评估指标
    metrics = calculate_metrics(returns)
    benchmark_metrics = calculate_metrics(benchmark_returns)
    
    # 打印评估指标
    print("\n策略评估指标:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n基准评估指标:")
    for metric, value in benchmark_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 绘制图表
    plot_returns(returns, benchmark_returns)
    plot_drawdown(returns)
    plot_monthly_returns(returns)
    
    # 保存结果
    results_path = os.path.join(save_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    with open(results_path, 'w') as f:
        yaml.dump({
            'strategy_metrics': metrics,
            'benchmark_metrics': benchmark_metrics
        }, f)
    
    print(f"\n分析结果已保存到: {results_path}")

def main():
    # 加载回测结果
    results_path = "models/saved/results_latest.yaml"
    results = load_backtest_results(results_path)
    
    # 分析策略
    analyze_strategy(results)

if __name__ == "__main__":
    main() 