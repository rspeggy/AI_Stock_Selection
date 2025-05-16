import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from custom_factors import load_and_prepare_data, create_custom_factors, load_factors, load_stock_data

# 定义所有行业股票
INDUSTRY_STOCKS = {
    'AI Healthcare': ["GH", "EXAS", "ILMN", "TDOC", "MDT"],
    'Fintech': ["PYPL", "COIN", "AFRM", "SOFI", "UPST"],
    'Clean Energy': ["TSLA", "ENPH", "FSLR", "PLUG", "NEE"],
    'Cloud and Big Data': ["AMZN", "MSFT", "GOOGL", "SNOW", "CRM"],
    'Semiconductor': ["NVDA", "AMD", "INTC", "ASML", "TSM"]
}

# 合并所有股票
ALL_STOCKS = []
for stocks in INDUSTRY_STOCKS.values():
    ALL_STOCKS.extend(stocks)

class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        初始化网络
        
        Args:
            state_dim (int): 状态维度
            action_dim (int): 动作维度
        """
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 动作标准差
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state (torch.Tensor): 状态
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (动作均值, 状态值)
        """
        features = self.feature_extractor(state)
        action_mean = self.actor(features)
        state_value = self.critic(features)
        if torch.isnan(action_mean).any() or torch.isnan(self.log_std).any():
            print("警告: 网络输出NaN!")
        return action_mean, state_value
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作
        
        Args:
            state (torch.Tensor): 状态
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (动作, 动作概率, 状态值)
        """
        action_mean, state_value = self.forward(state)
        action_std = torch.exp(self.log_std)
        
        # 创建正态分布
        dist = Normal(action_mean, action_std)
        
        # 采样动作
        action = dist.rsample()
        
        # 计算动作概率
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, action_log_prob, state_value

class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 K_epochs: int = 10):
        """
        初始化训练器
        
        Args:
            state_dim (int): 状态维度
            action_dim (int): 动作维度
            lr (float): 学习率
            gamma (float): 折扣因子
            eps_clip (float): PPO裁剪参数
            K_epochs (int): 每批数据的训练轮数
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # 创建网络
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 创建旧策略网络
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        选择动作
        
        Args:
            state (torch.Tensor): 状态
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (动作, 动作概率, 状态值)
        """
        with torch.no_grad():
            action, action_log_prob, state_value = self.policy_old.get_action(state)
        return action, action_log_prob, state_value
    
    def update(self, memory: List[Dict]) -> Dict:
        """
        更新策略
        
        Args:
            memory (List[Dict]): 经验回放
            
        Returns:
            Dict: 训练统计信息
        """
        # 提取经验
        states = torch.FloatTensor([m['state'] for m in memory])
        actions = torch.FloatTensor([m['action'] for m in memory])
        old_log_probs = torch.FloatTensor([m['log_prob'] for m in memory])
        rewards = torch.FloatTensor([m['reward'] for m in memory])
        next_states = torch.FloatTensor([m['next_state'] for m in memory])
        dones = torch.FloatTensor([m['done'] for m in memory])
        
        # 计算优势
        with torch.no_grad():
            _, _, next_values = self.policy_old.get_action(next_states)
            _, _, values = self.policy_old.get_action(states)
            
            # 计算TD误差
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            
            # 计算优势
            advantages = torch.zeros_like(rewards)
            running_add = 0
            for t in reversed(range(len(rewards))):
                running_add = deltas[t] + self.gamma * running_add * (1 - dones[t])
                advantages[t] = running_add
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        for _ in range(self.K_epochs):
            # 获取新策略的动作概率和状态值
            _, new_log_probs, new_values = self.policy.get_action(states)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算PPO目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # 计算策略损失和价值损失
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (new_values - (values + advantages)).pow(2).mean()
            
            # 总损失
            loss = policy_loss + value_loss
            
            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': loss.item()
        }

class HealthcareTradingEnv:
    """医疗健康股票交易环境"""
    
    def __init__(self,
                 price_data: pd.DataFrame,
                 factor_data: pd.DataFrame = None,
                 initial_balance: float = 1000000,
                 transaction_cost: float = 0.001):
        """
        初始化环境
        
        Args:
            price_data (pd.DataFrame): 股票价格数据
            factor_data (pd.DataFrame): 因子数据
            initial_balance (float): 初始资金
            transaction_cost (float): 交易成本
        """
        self.price_data = price_data.replace(0, 1e-6).fillna(method='ffill').fillna(method='bfill')
        self.factor_data = factor_data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            np.ndarray: 初始状态
        """
        self.balance = self.initial_balance
        self.positions = np.zeros(len(self.price_data.columns))
        self.current_step = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        获取当前状态
        
        Returns:
            np.ndarray: 状态向量
        """
        # 获取当前价格
        current_prices = self.price_data.iloc[self.current_step].values
        
        # 计算持仓价值
        position_values = self.positions * current_prices
        
        # 计算总资产
        total_value = self.balance + np.sum(position_values)
        
        # 计算持仓比例
        position_ratios = position_values / total_value
        
        # 获取当前因子值
        if self.factor_data is not None:
            current_factors = self.factor_data.iloc[self.current_step].values
        else:
            current_factors = np.array([])
        
        # 组合状态向量
        state = np.concatenate([
            [self.balance / self.initial_balance],  # 归一化余额
            position_ratios,  # 持仓比例
            current_prices / current_prices.mean(),  # 归一化价格
            current_factors  # 因子值
        ])
        
        if np.isnan(state).any() or np.isinf(state).any():
            print("警告: state中含有NaN或Inf!", state)
        
        if np.isnan(state).any() or np.isinf(state).any():
            print("警告: 环境state含有NaN/Inf!", state)
        
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action (np.ndarray): 动作向量
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict]: (新状态, 奖励, 是否结束, 信息)
        """
        # 获取当前价格
        current_prices = self.price_data.iloc[self.current_step].values
        
        # 计算目标持仓
        target_positions = action * self.balance / current_prices
        
        # 计算交易成本
        transaction_amount = np.abs(target_positions - self.positions) * current_prices
        transaction_cost = np.sum(transaction_amount) * self.transaction_cost
        
        # 更新持仓和余额
        self.positions = target_positions
        self.balance -= transaction_cost
        
        # 计算奖励（收益率）
        if self.current_step > 0:
            prev_prices = self.price_data.iloc[self.current_step - 1].values
            returns = (current_prices - prev_prices) / prev_prices
            reward = np.sum(self.positions * returns)
        else:
            reward = 0
        
        # 更新步数
        self.current_step += 1

        # 检查是否结束
        done = self.current_step >= len(self.price_data)

        # 获取新状态（只有没结束时才取，否则返回上一个state或全0）
        if not done:
            next_state = self._get_state()
        else:
            # 可以返回上一个state，或全0
            next_state = np.zeros_like(self._get_state())

        # 信息
        info = {
            'balance': self.balance,
            'positions': self.positions,
            'transaction_cost': transaction_cost
        }

        if np.isnan(next_state).any() or np.isinf(next_state).any():
            print("警告: state中含有NaN或Inf!", next_state)

        return next_state, reward, done, info

def train_rl_agent(price_data: pd.DataFrame,
                  state_dim: int,
                  action_dim: int,
                  factor_data: pd.DataFrame = None,
                  n_episodes: int = 1000,
                  max_steps: int = 1000,
                  save_dir: str = "models/saved") -> Dict:
    """
    训练RL代理
    
    Args:
        price_data (pd.DataFrame): 价格数据
        state_dim (int): 状态维度
        action_dim (int): 动作维度
        factor_data (pd.DataFrame): 因子数据
        n_episodes (int): 训练轮数
        max_steps (int): 每轮最大步数
        save_dir (str): 保存目录
        
    Returns:
        Dict: 训练结果
    """
    # 创建环境
    env = HealthcareTradingEnv(price_data, factor_data)
    
    # 创建训练器
    trainer = PPOTrainer(state_dim, action_dim)
    
    # 训练记录
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': []
    }
    
    # 填充NaN
    price_data = price_data.replace(0, 1e-6).fillna(method='ffill').fillna(method='bfill')
    factor_data = factor_data.fillna(0)
    
    # 训练循环
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        memory = []
        
        for step in range(max_steps):
            # 选择动作
            action, log_prob, value = trainer.select_action(torch.FloatTensor(state))
            action = action.numpy()
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            memory.append({
                'state': state,
                'action': action,
                'log_prob': log_prob,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            # 更新状态和统计信息
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # 更新策略
        update_stats = trainer.update(memory)
        
        # 记录统计信息
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_lengths'].append(episode_length)
        training_stats['policy_losses'].append(update_stats['policy_loss'])
        training_stats['value_losses'].append(update_stats['value_loss'])
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"Average Reward: {np.mean(training_stats['episode_rewards'][-10:]):.2f}")
            print(f"Average Length: {np.mean(training_stats['episode_lengths'][-10:]):.2f}")
            print(f"Policy Loss: {update_stats['policy_loss']:.4f}")
            print(f"Value Loss: {update_stats['value_loss']:.4f}")
            print()
    
    # 保存模型
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"rl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save(trainer.policy.state_dict(), model_path)
    
    return {
        'training_stats': training_stats,
        'model_path': model_path
    }

if __name__ == "__main__":
    # 使用所有股票
    symbols = ALL_STOCKS
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"\n开始加载数据 - 日期范围: {start_date} 到 {end_date}")
    print(f"处理股票数量: {len(symbols)}")
    print(f"股票列表: {symbols}")
    
    # 直接读取因子数据
    all_factors = {}
    price_data = pd.DataFrame()
    
    for symbol in symbols:
        print(f"\n处理股票: {symbol}")
        # 读取因子数据
        factors = load_factors(symbol, start_date, end_date)
        if factors is not None and not factors.empty:
            print(f"成功加载因子数据 - 形状: {factors.shape}")
            all_factors[symbol] = factors
            
            # 读取价格数据
            stock_data = load_stock_data(symbol)
            if stock_data is not None and not stock_data.empty:
                # 过滤日期范围
                if start_date:
                    stock_data = stock_data[stock_data.index >= start_date]
                if end_date:
                    stock_data = stock_data[stock_data.index <= end_date]
                
                if not stock_data.empty:
                    print(f"成功加载价格数据 - 形状: {stock_data.shape}")
                    price_data[symbol] = stock_data['Close']
                else:
                    print(f"警告: 过滤后的价格数据为空 - {symbol}")
            else:
                print(f"错误: 无法加载价格数据 - {symbol}")
        else:
            print(f"错误: 无法加载因子数据 - {symbol}")
    
    # 合并因子数据
    factor_data = pd.DataFrame()
    for symbol in symbols:
        if symbol in all_factors:
            for factor_name in all_factors[symbol].columns:
                factor_data[f"{symbol}_{factor_name}"] = all_factors[symbol][factor_name]
    
    # 确保价格数据和因子数据的日期对齐
    common_dates = price_data.index.intersection(factor_data.index)
    if len(common_dates) == 0:
        raise ValueError("价格数据和因子数据没有共同的日期")
        
    price_data = price_data.loc[common_dates]
    factor_data = factor_data.loc[common_dates]
    
    # 计算状态和动作维度
    state_dim = 1 + len(symbols) + len(symbols) + len(factor_data.columns)  # 余额 + 持仓比例 + 价格 + 因子
    action_dim = len(symbols)  # 每个股票的持仓比例
    
    print("\n数据准备完成:")
    print(f"价格数据形状: {price_data.shape}")
    print(f"因子数据形状: {factor_data.shape}")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"数据日期范围: {price_data.index.min()} 到 {price_data.index.max()}")
    
    # 填充NaN
    price_data = price_data.replace(0, 1e-6).fillna(method='ffill').fillna(method='bfill')
    factor_data = factor_data.fillna(0)
    
    # 训练RL代理
    result = train_rl_agent(price_data, state_dim, action_dim, factor_data)
    
    # 打印训练结果
    print("\n训练完成!")
    print(f"模型已保存到: {result['model_path']}")
    print(f"平均奖励: {np.mean(result['training_stats']['episode_rewards'][-100:]):.2f}")
    print(f"平均回合长度: {np.mean(result['training_stats']['episode_lengths'][-100:]):.2f}") 