import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class ManualOptimizer:
    def __init__(self, returns: pd.DataFrame, risk_free_rate=0.02, lr=0.001, max_iter=500, tol=1e-6):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
        self.n_assets = len(returns.columns)
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def portfolio_performance(self, weights):
        ret = np.dot(weights, self.mean_returns)
        vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        sharpe = (ret - self.risk_free_rate) / vol
        return ret, vol, sharpe

    def negative_sharpe(self, w):
        r, v, _ = self.portfolio_performance(w)
        return -(r - self.risk_free_rate) / v

    def grad_negative_sharpe(self, w):
        ret = np.dot(w, self.mean_returns)
        vol = np.sqrt(w.T @ self.cov_matrix @ w)
        grad_ret = self.mean_returns
        grad_vol = (self.cov_matrix @ w) / vol
        grad_sharpe = (grad_ret * vol - (ret - self.risk_free_rate) * grad_vol) / (vol ** 2)
        return -grad_sharpe

    def project_simplex(self, weights):
        sorted_w = np.sort(weights)[::-1]
        tmp_sum = 0
        for i in range(len(weights)):
            tmp_sum += sorted_w[i]
            t = (tmp_sum - 1) / (i + 1)
            if i == len(weights) - 1 or sorted_w[i + 1] <= t:
                theta = t
                break
        return np.maximum(weights - theta, 0)

    def optimize_with_gradient(self):
        w = np.ones(self.n_assets) / self.n_assets
        for _ in range(self.max_iter):
            grad = self.grad_negative_sharpe(w)
            grad = grad / (np.linalg.norm(grad) + 1e-8)
            w_new = w - self.lr * grad
            w_new = self.project_simplex(w_new)
            if np.linalg.norm(w_new - w) < self.tol:
                break
            w = w_new
        r, v, s = self.portfolio_performance(w)
        return w, r, v, s

    def optimize_with_interior_point(self, mu=10, t_init=1.0):
        w = np.ones(self.n_assets) / self.n_assets
        t = t_init
        eps = 1e-8
        for _ in range(self.max_iter):
            grad = self.grad_negative_sharpe(w) * t - 1 / (w + eps)
            w_new = w - self.lr * grad
            w_new = self.project_simplex(w_new)
            w_new = np.clip(w_new, eps, 1)  # 防止为0
            if np.linalg.norm(w_new - w) < self.tol:
                if t > 100:
                    break
                t *= mu
            w = w_new
        r, v, s = self.portfolio_performance(w)
        return w, r, v, s

    def optimize_with_kkt(self):
        # Simplified primal-dual updates
        w = np.ones(self.n_assets) / self.n_assets
        lambd = 0.0  # dual var for sum(w)=1
        for _ in range(self.max_iter):
            grad = self.grad_negative_sharpe(w)
            grad = grad + lambd  # include dual
            grad = grad / (np.linalg.norm(grad) + 1e-8)
            w_new = w - self.lr * grad
            w_new = np.maximum(w_new, 0)  # enforce non-negativity
            lambd = lambd + self.lr * (np.sum(w_new) - 1)  # update dual
            if np.linalg.norm(w_new - w) < self.tol:
                break
            w = w_new
        r, v, s = self.portfolio_performance(w)
        return w, r, v, s

    def optimize_with_scipy(self, max_weight=0.3, min_weight=0.0):
        def neg_sharpe(w):
            return self.negative_sharpe(w)
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        w0 = np.ones(self.n_assets) / self.n_assets
        result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons)
        w_opt = result.x
        r, v, s = self.portfolio_performance(w_opt)
        return w_opt, r, v, s

    def optimize_portfolio(self, method='gradient', max_weight=0.3, min_weight=0.0):
        if method == 'gradient':
            w, r, v, s = self.optimize_with_gradient()
        elif method == 'interior_point':
            w, r, v, s = self.optimize_with_interior_point()
        elif method == 'kkt':
            w, r, v, s = self.optimize_with_kkt()
        elif method == 'scipy':
            w, r, v, s = self.optimize_with_scipy(max_weight=max_weight, min_weight=min_weight)
        else:
            raise ValueError(f"Unknown method: {method}")
        return {
            'weights': dict(zip(self.returns.columns, w)),
            'expected_return': r,
            'volatility': v,
            'sharpe_ratio': s
        }

if __name__ == "__main__":
    symbols = ["GH", "EXAS", "ILMN", "TDOC"]
    data_dir = "data/raw"
    data_frames = []
    for symbol in symbols:
        try:
            # 读取第一行作为列名，并将第一个列名替换为'Date'
            columns = pd.read_csv(f"{data_dir}/{symbol}.csv", nrows=0).columns
            columns = ['Date'] + list(columns[1:])
            # 跳过前两行（Price和Ticker行）和第三行（空行）
            df = pd.read_csv(f"{data_dir}/{symbol}.csv", skiprows=3, names=columns)
            # 将Date列设置为索引
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            data_frames.append(df['Close'].rename(symbol))
        except Exception as e:
            print(f"加载 {symbol} 数据时出错: {str(e)}")
    
    if not data_frames:
        raise ValueError("没有成功加载任何股票数据")
    
    # 合并成一个DataFrame，只保留所有股票都存在的日期
    adj_close = pd.concat(data_frames, axis=1, join='inner')
    print("adj_close shape:", adj_close.shape)
    print("adj_close head:\n", adj_close.head())

    # 计算收益率
    returns = adj_close.pct_change().dropna()
    print("returns shape:", returns.shape)
    print("returns head:\n", returns.head())

    optimizer = ManualOptimizer(returns)

    methods = ['gradient', 'interior_point', 'kkt', 'scipy']
    expected_returns = []
    volatilities = []
    sharpe_ratios = []

    for method in methods:
        print(f"\n=== 优化方法: {method} ===")
        if method == 'scipy':
            result = optimizer.optimize_portfolio(method=method, max_weight=0.3, min_weight=0.0)
        else:
            result = optimizer.optimize_portfolio(method=method)
        print("权重:")
        for sym, weight in result['weights'].items():
            print(f"{sym}: {weight:.2%}")
        print(f"预期收益率: {result['expected_return']:.2%}")
        print(f"波动率: {result['volatility']:.2%}")
        print(f"夏普比率: {result['sharpe_ratio']:.2f}")
        expected_returns.append(result['expected_return'])
        volatilities.append(result['volatility'])
        sharpe_ratios.append(result['sharpe_ratio'])

    x = np.arange(len(methods))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.bar(x, expected_returns, color='skyblue')
    plt.xticks(x, methods)
    plt.ylabel('Expected Return')
    plt.title('Expected Return Comparison')

    plt.subplot(1, 3, 2)
    plt.bar(x, volatilities, color='orange')
    plt.xticks(x, methods)
    plt.ylabel('Volatility')
    plt.title('Volatility Comparison')

    plt.subplot(1, 3, 3)
    plt.bar(x, sharpe_ratios, color='green')
    plt.xticks(x, methods)
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio Comparison')

    plt.tight_layout()
    plt.show()
