import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set font for plots
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.family'] = 'Arial'  # 改成你喜欢的字体，比如 'Times New Roman'、'Calibri'、'SimHei' (中文黑体)

class ManualOptimizer:
    def __init__(self, returns: pd.DataFrame, risk_free_rate=0.02, lr=0.001, max_iter=2000, tol=1e-6):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
        self.n_assets = len(returns.columns)
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.history = {}  # 用于存储优化历史

    def reset_history(self):
        """重置历史记录"""
        self.history = {
            'objective': [],
            'weights': [],
            'iterations': []
        }

    def portfolio_performance(self, weights):
        """Calculate portfolio performance"""
        ret = np.dot(weights, self.mean_returns)
        vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        sharpe = (ret - self.risk_free_rate) / vol if vol != 0 else np.nan
        return ret, vol, sharpe

    def negative_sharpe(self, w):
        """Calculate negative Sharpe ratio"""
        r, v, _ = self.portfolio_performance(w)
        if np.isnan(r) or np.isnan(v) or v == 0:
            return np.inf
        return -(r - self.risk_free_rate) / v

    def grad_negative_sharpe(self, w):
        """Calculate gradient of negative Sharpe ratio"""
        ret = np.dot(w, self.mean_returns)
        vol = np.sqrt(w.T @ self.cov_matrix @ w)
        grad_ret = self.mean_returns
        grad_vol = (self.cov_matrix @ w) / vol
        grad_sharpe = (grad_ret * vol - (ret - self.risk_free_rate) * grad_vol) / (vol ** 2)
        return -grad_sharpe

    def project_simplex(self, weights):
        """Project weights onto simplex"""
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
        """Optimize using gradient descent"""
        self.reset_history()
        w = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        best_sharpe = -np.inf
        best_w = w.copy()
        
        for i in range(self.max_iter):
            grad = self.grad_negative_sharpe(w)
            if np.any(np.isnan(grad)):
                break
                
            grad = grad / (np.linalg.norm(grad) + 1e-8)
            w_new = w - self.lr * grad
            w_new = self.project_simplex(w_new)
            
            # 记录历史
            self.history['objective'].append(self.negative_sharpe(w_new))
            self.history['weights'].append(w_new.copy())
            self.history['iterations'].append(i)
            
            # Check if new weights produce better Sharpe ratio
            _, _, sharpe = self.portfolio_performance(w_new)
            if not np.isnan(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_w = w_new.copy()
            
            if np.linalg.norm(w_new - w) < self.tol:
                break
            w = w_new
            
        return best_w, *self.portfolio_performance(best_w)

    def optimize_with_interior_point(self, mu=10, t_init=1.0):
        """Optimize using interior point method"""
        self.reset_history()
        w = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        t = t_init
        eps = 1e-5  # 增大eps
        best_sharpe = -np.inf
        best_w = w.copy()
        
        for i in range(self.max_iter):
            grad = self.grad_negative_sharpe(w)
            if np.any(np.isnan(grad)):
                break
                
            grad = grad * t - 1 / (w + eps)
            w_new = w - self.lr * grad
            w_new = self.project_simplex(w_new)
            w_new = np.clip(w_new, eps, 1)
            obj = self.negative_sharpe(w_new)
            # 只记录合理范围的目标函数
            if np.isfinite(obj) and abs(obj) < 10:
                self.history['objective'].append(obj)
                self.history['weights'].append(w_new.copy())
                self.history['iterations'].append(i)
            else:
                break  # 目标函数异常，提前终止
            
            _, _, sharpe = self.portfolio_performance(w_new)
            if not np.isnan(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_w = w_new.copy()
            
            if np.linalg.norm(w_new - w) < self.tol:
                if t > 100:
                    break
                t *= mu
            w = w_new
            
        return best_w, *self.portfolio_performance(best_w)

    def optimize_with_kkt(self):
        """Optimize using KKT conditions"""
        self.reset_history()
        w = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        lambd = 0.0
        best_sharpe = -np.inf
        best_w = w.copy()
        
        for i in range(self.max_iter):
            grad = self.grad_negative_sharpe(w)
            if np.any(np.isnan(grad)):
                break
                
            grad = grad + lambd
            grad = grad / (np.linalg.norm(grad) + 1e-8)
            w_new = w - self.lr * grad
            w_new = np.maximum(w_new, 0)
            w_new = self.project_simplex(w_new)
            
            # 记录历史
            self.history['objective'].append(self.negative_sharpe(w_new))
            self.history['weights'].append(w_new.copy())
            self.history['iterations'].append(i)
            
            # Check if new weights produce better Sharpe ratio
            _, _, sharpe = self.portfolio_performance(w_new)
            if not np.isnan(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_w = w_new.copy()
            
            if np.linalg.norm(w_new - w) < self.tol:
                break
            w = w_new
            lambd = lambd + self.lr * (np.sum(w_new) - 1)
            
        return best_w, *self.portfolio_performance(best_w)

    def optimize_with_scipy(self, max_weight=0.3, min_weight=0.0):
        """Optimize using SciPy"""
        def neg_sharpe(w):
            return self.negative_sharpe(w)
            
        def weight_sum(w):
            return np.sum(w) - 1.0
            
        cons = [
            {'type': 'eq', 'fun': weight_sum}
        ]
        
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        w0 = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        
        result = minimize(
            neg_sharpe, 
            w0, 
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-8, 'disp': False}
        )
        
        if not result.success:
            logging.warning(f"SciPy optimization did not converge: {result.message}")
            return w0, *self.portfolio_performance(w0)
            
        w_opt = result.x
        # Ensure weights sum to 1
        w_opt = w_opt / np.sum(w_opt)
        return w_opt, *self.portfolio_performance(w_opt)

    def optimize_portfolio(self, method='gradient', max_weight=0.3, min_weight=0.0):
        """Optimize portfolio"""
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
            
        # Ensure weights sum to 1
        w = w / np.sum(w)
        
        return {
            'weights': dict(zip(self.returns.columns, w)),
            'expected_return': r,
            'volatility': v,
            'sharpe_ratio': s
        }

    def plot_convergence_history(self, method_name, industry_name):
        """绘制优化收敛历史"""
        if not self.history['objective']:
            logging.warning(f"No convergence history available for {method_name}")
            return
            
        plt.figure(figsize=(10, 6))
        y = np.array(self.history['objective'])
        # 过滤：大于1或小于-1的设置为nan
        y[(y > 1) | (y < -1)] = np.nan
        x = np.array(self.history['iterations'])
        mask = np.isfinite(y)
        plt.plot(x[mask], y[mask], label='Objective Value (-Sharpe Ratio)', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value (-Sharpe Ratio)')
        plt.title(f'Convergence History - {method_name}\n{industry_name}')
        plt.grid(True)
        plt.legend()
        
        # 保存图片
        filename = f'results/convergence_{method_name}_{industry_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename)
        plt.close()
        logging.info(f"Saved convergence plot to {filename}")

def get_available_stocks(data_dir="data/raw"):
    """Get available stock data files"""
    available_stocks = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            available_stocks.append(file.replace('.csv', ''))
    return available_stocks

def load_stock_data(symbols, data_dir="data/raw", start_date=None, end_date=None):
    """Load stock data"""
    data_frames = []
    
    for symbol in symbols:
        try:
            # Read first row as column names and replace first column name with 'Date'
            columns = pd.read_csv(f"{data_dir}/{symbol}.csv", nrows=0).columns
            columns = ['Date'] + list(columns[1:])
            # Skip first two rows (Price and Ticker rows) and third row (empty row)
            df = pd.read_csv(f"{data_dir}/{symbol}.csv", skiprows=3, names=columns)
            # Set Date column as index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Filter by date range if provided
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
                
            data_frames.append(df['Close'].rename(symbol))
            logging.info(f"Successfully loaded data for {symbol}")
        except Exception as e:
            logging.error(f"Error loading data for {symbol}: {str(e)}")
    
    if not data_frames:
        raise ValueError("No stock data was successfully loaded")
    
    # Merge into a single DataFrame, keeping only dates where all stocks exist
    adj_close = pd.concat(data_frames, axis=1, join='inner')
    logging.info(f"Successfully merged data, total of {len(adj_close.columns)} stocks")
    return adj_close

def analyze_industry(industry_name, symbols, returns, methods):
    """Analyze a single industry"""
    logging.info(f"\n===== Industry: {industry_name} =====")
    
    # Check data completeness
    available_symbols = [s for s in symbols if s in returns.columns]
    if not available_symbols:
        logging.error(f"No available stock data for industry {industry_name}")
        return {method: None for method in methods}
    
    if len(available_symbols) < len(symbols):
        logging.warning(f"Some stock data unavailable for industry {industry_name}: {set(symbols) - set(available_symbols)}")
    
    industry_returns = returns[available_symbols]
    # Ensure data type is float64
    industry_returns = industry_returns.astype(np.float64)
    optimizer = ManualOptimizer(industry_returns)
    results = {}
    
    for method in methods:
        try:
            result = optimizer.optimize_portfolio(method=method)
            logging.info(f"\n=== Optimization Method: {method} ===")
            for sym, weight in result['weights'].items():
                logging.info(f"{sym}: {weight:.2%}")
            logging.info(f"Expected Return: {result['expected_return']:.2%}")
            logging.info(f"Volatility: {result['volatility']:.2%}")
            logging.info(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            
            # 绘制收敛历史
            optimizer.plot_convergence_history(method, industry_name)
            
            results[method] = result
        except Exception as e:
            logging.error(f"Error optimizing with {method} method: {str(e)}")
            results[method] = None
    
    return results

def plot_results(results, industry_name, methods):
    """Plot results for a single industry"""
    metrics = ['expected_return', 'volatility', 'sharpe_ratio']
    metric_names = {
        'expected_return': 'Return Rate',
        'volatility': 'Volatility',
        'sharpe_ratio': 'Sharpe Ratio'
    }
    
    # Collect data
    data = {metric: [] for metric in metrics}
    labels = []
    
    for method in methods:
        if results[method] is not None:
            result = results[method]
            if not np.isnan(result['expected_return']):
                for metric in metrics:
                    if metric in ['expected_return', 'volatility']:
                        # Multiply by 100 for percentage display
                        data[metric].append(result[metric] * 100)
                    else:
                        data[metric].append(result[metric])
                labels.append(f"{method}")
    
    if not any(data.values()):
        logging.warning(f"No available data for plotting industry {industry_name}")
        return
    
    # Create subplots
    plt.figure(figsize=(9, 3))
    x = np.arange(len(labels))
    width = 0.25
    
    # Plot three metrics
    plt.subplot(1, 3, 1)
    plt.bar(x, data['expected_return'], width, color='#91c2d8')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Return Rate (%)')
    plt.title(f'{industry_name} - Return Rate')
    plt.ylim(0, 0.25)  # 这里设置Return Rate的y轴范围
    
    plt.subplot(1, 3, 2)
    plt.bar(x, data['volatility'], width, color='#f7932d')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Volatility (%)')
    plt.title(f'{industry_name} - Volatility')
    plt.ylim(0, 7)  # 这里设置Volatility的y轴范围
    
    plt.subplot(1, 3, 3)
    plt.bar(x, data['sharpe_ratio'], width, color='#fcd82c')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Sharpe Ratio')
    plt.title(f'{industry_name} - Sharpe Ratio')
    plt.ylim(-1, 0)  # 这里设置Sharpe Ratio的y轴范围
    
    plt.tight_layout()
    filename = f'results/optimization_comparison_{industry_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename)
    logging.info(f"Saved plot for {industry_name} to {filename}")
    plt.close()

def plot_industry_comparison(all_results, industry_groups, methods):
    """Plot industry comparison chart"""
    # Collect best results for all industries (using scipy method)
    industry_metrics = {
        'expected_return': [],
        'volatility': [],
        'sharpe_ratio': []
    }
    industries = []
    
    for industry in industry_groups.keys():
        if 'scipy' in all_results[industry] and all_results[industry]['scipy'] is not None:
            result = all_results[industry]['scipy']
            if not np.isnan(result['expected_return']):
                industries.append(industry)
                # Multiply by 100 for percentage display
                industry_metrics['expected_return'].append(result['expected_return'] * 100)
                industry_metrics['volatility'].append(result['volatility'] * 100)
                industry_metrics['sharpe_ratio'].append(result['sharpe_ratio'])
    
    if not industries:
        logging.warning("No available industry data for comparison")
        return
    
    # Create subplots
    plt.figure(figsize=(9, 3))
    x = np.arange(len(industries))
    width = 0.25
    
    # Plot three metrics
    plt.subplot(1, 3, 1)
    plt.bar(x, industry_metrics['expected_return'], width, color='#0975b3')
    plt.xticks(x, industries, rotation=45, ha='right')
    plt.ylabel('Return Rate (%)')
    plt.title('Industry Return Rate')
    
    plt.subplot(1, 3, 2)
    plt.bar(x, industry_metrics['volatility'], width, color='#bbe7fc')
    plt.xticks(x, industries, rotation=45, ha='right')
    plt.ylabel('Volatility (%)')
    plt.title('Industry Volatility')
    
    plt.subplot(1, 3, 3)
    plt.bar(x, industry_metrics['sharpe_ratio'], width, color='#f57823')
    plt.xticks(x, industries, rotation=45, ha='right')
    plt.ylabel('Sharpe Ratio')
    plt.title('Industry Sharpe Ratio')
    
    plt.tight_layout()
    plt.savefig('results/industry_comparison.png')
    logging.info("Saved industry comparison plot to results/industry_comparison.png")
    plt.close()

def analyze_industry_performance(all_results, industry_groups):
    """Analyze industry performance and generate report"""
    performance_data = []
    
    for industry in industry_groups.keys():
        if 'scipy' in all_results[industry] and all_results[industry]['scipy'] is not None:
            result = all_results[industry]['scipy']
            if not np.isnan(result['expected_return']):
                performance_data.append({
                    'Industry': industry,
                    'Expected Return': result['expected_return'],
                    'Volatility': result['volatility'],
                    'Sharpe Ratio': result['sharpe_ratio'],
                    'Risk-Adjusted Return': result['expected_return'] / result['volatility'] if result['volatility'] != 0 else np.nan
                })
    
    if performance_data:
        # Create DataFrame and sort
        df = pd.DataFrame(performance_data)
        
        # Sort by Sharpe ratio
        df_sharpe = df.sort_values('Sharpe Ratio', ascending=False)
        logging.info("\n=== Industry Performance Ranking (by Sharpe Ratio) ===")
        logging.info(df_sharpe.to_string(index=False))
        
        # Sort by risk-adjusted return
        df_risk_adj = df.sort_values('Risk-Adjusted Return', ascending=False)
        logging.info("\n=== Industry Performance Ranking (by Risk-Adjusted Return) ===")
        logging.info(df_risk_adj.to_string(index=False))
        
        # Save detailed report
        df.to_csv('results/industry_performance_report.csv', index=False, encoding='utf-8-sig')
        logging.info("\nIndustry performance report saved to results/industry_performance_report.csv")

if __name__ == "__main__":
    # Create results directory
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Stock industry groups - only use stocks from v1 version
    industry_groups = {
        'AI Healthcare': ["GH", "EXAS", "ILMN", "TDOC", "MDT"],
        'Fintech': ["PYPL", "COIN", "AFRM", "SOFI", "UPST"],
        'Clean Energy': ["TSLA", "ENPH", "FSLR", "PLUG", "NEE"],
        'Cloud and Big Data': ["AMZN", "MSFT", "GOOGL", "SNOW", "CRM"],
        'Semiconductor': ["NVDA", "AMD", "INTC", "ASML", "TSM"],
        'Customize':["ENPH", "PYPL", "GH", "SNOW", "PLUG", "FSLR"]
    }

    # Set time ranges for training and testing
    train_start_date = "2019-01-01"
    train_end_date = "2022-12-31"
    test_start_date = "2023-01-01"
    test_end_date = "2024-12-31"
    
    # Optimization methods
    methods = ['gradient', 'interior_point', 'kkt', 'scipy']
    
    # Store results for all industries
    train_results = {}
    test_results = {}
    
    # Analyze each industry for both training and testing periods
    for period_name, (start_date, end_date) in [("Training", (train_start_date, train_end_date)), 
                                               ("Testing", (test_start_date, test_end_date))]:
        logging.info(f"\n===== Starting {period_name} Period Analysis =====")
        current_results = {}
        
        for industry, symbols in industry_groups.items():
            logging.info(f"\n===== Industry: {industry} =====")
            
            # Load data for this industry
            logging.info(f"Loading stock data for {industry} industry...")
            try:
                adj_close = load_stock_data(symbols, start_date=start_date, end_date=end_date)
                returns = adj_close.pct_change().dropna()
                
                # Print data information
                logging.info(f"adj_close shape: {adj_close.shape}")
                logging.info(f"adj_close head:\n{adj_close.head()}")
                logging.info(f"returns shape: {returns.shape}")
                logging.info(f"returns head:\n{returns.head()}")
                
                industry_returns = returns
                optimizer = ManualOptimizer(industry_returns)
                results = {}
                
                for method in methods:
                    try:
                        logging.info(f"\n=== Optimization Method: {method} ===")
                        if method == 'scipy':
                            result = optimizer.optimize_portfolio(method=method, max_weight=0.3, min_weight=0.0)
                        else:
                            result = optimizer.optimize_portfolio(method=method)
                        
                        logging.info("Weights:")
                        for sym, weight in result['weights'].items():
                            logging.info(f"{sym}: {weight:.2%}")
                        logging.info(f"Expected Return: {result['expected_return']:.2%}")
                        logging.info(f"Volatility: {result['volatility']:.2%}")
                        logging.info(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                        
                        # 绘制收敛历史
                        optimizer.plot_convergence_history(method, f"{industry}_{period_name}")
                        
                        results[method] = result
                    except Exception as e:
                        logging.error(f"Error optimizing with {method} method: {str(e)}")
                        results[method] = None
                
                # Plot results for each industry
                plot_results(results, f"{industry}_{period_name}", methods)
                
                # Save results to CSV
                summary_data = []
                for method in methods:
                    if results[method] is not None:
                        result = results[method]
                        if not np.isnan(result['expected_return']):
                            summary_data.append({
                                'Industry': industry,
                                'Period': period_name,
                                'Optimization Method': method,
                                'Expected Return': result['expected_return'],
                                'Volatility': result['volatility'],
                                'Sharpe Ratio': result['sharpe_ratio']
                            })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_csv(f'results/optimization_summary_{industry.lower().replace(" ", "_")}_{period_name.lower()}.csv', 
                                    index=False, encoding='utf-8-sig')
                    logging.info(f"\n{industry} industry analysis completed! Results saved to results directory.")
                    current_results[industry] = results
                else:
                    logging.error(f"No valid optimization results generated for {industry} industry")
                    
            except Exception as e:
                logging.error(f"Error processing {industry} industry: {str(e)}")
                continue
        
        # Store results for this period
        if period_name == "Training":
            train_results = current_results
        else:
            test_results = current_results
            
        # Industry comparison for this period
        if current_results:
            logging.info(f"\n===== Starting Industry Comparison for {period_name} Period =====")
            plot_industry_comparison(current_results, industry_groups, methods)
            analyze_industry_performance(current_results, industry_groups)
            
    # Compare training and testing results
    logging.info("\n===== Comparing Training and Testing Results =====")
    for industry in industry_groups.keys():
        if industry in train_results and industry in test_results:
            train_result = train_results[industry].get('interior_point')
            test_result = test_results[industry].get('interior_point')
            
            if train_result and test_result:
                logging.info(f"\nIndustry: {industry}")
                logging.info("Training Period:")
                logging.info(f"Expected Return: {train_result['expected_return']:.2%}")
                logging.info(f"Volatility: {train_result['volatility']:.2%}")
                logging.info(f"Sharpe Ratio: {train_result['sharpe_ratio']:.2f}")
                logging.info("\nTesting Period:")
                logging.info(f"Expected Return: {test_result['expected_return']:.2%}")
                logging.info(f"Volatility: {test_result['volatility']:.2%}")
                logging.info(f"Sharpe Ratio: {test_result['sharpe_ratio']:.2f}")
