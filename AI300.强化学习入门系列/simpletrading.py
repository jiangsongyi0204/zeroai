import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import random

ACTION_DIM = 3
TIME_RANGE = 200
WIN_SIZE = 10
STATE_DIM = WIN_SIZE * 5 + 2 
EPOCH_LOOP = 200

# 模型
class SimpleTradingDNN(nn.Module):
    def __init__(self):
        super(SimpleTradingDNN, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, STATE_DIM * 2)  # 输入层到隐藏层
        self.fc2 = nn.Linear(STATE_DIM * 2, 3)  # 隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层，使用ReLU激活函数
        x = self.fc2(x)  # 输出层
        return x

class SimpleTradingEnv(gym.Env):
    """
        股票交易强化学习环境
        动作空间: 0=买入, 1=卖出, 2=持仓
        状态空间: [历史价格窗口, 是否持仓, 账户余额]
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int = WIN_SIZE,
                 initial_capital: float = 10000.0,
                 transaction_fee: float = 0.001  # 交易额的千分之一
                 ):
        """
        初始化环境
        :param df: 包含OHLCV的DataFrame (Open, High, Low, Close, Volume)
        :param window_size: 观测历史窗口大小
        :param initial_capital: 初始资金
        :param transaction_fee: 交易手续费率
        """
        super(SimpleTradingEnv, self).__init__()

        # 数据参数
        self.df = df.dropna().reset_index(drop=True)
        self.window_size = window_size
        self.current_step = None            # 与时间下标对应
        self.current_date = None

        # 交易参数
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee

        # 状态信息
        self.capital = None         # 当前账户市值
        self.cash = None            # 当前现金
        self.market_value = None    # 当前市值
        self.shares = None          # 当前持仓份额
        self.history = []           # 记录交易历史

        # 定义动作空间
        self.action_space = spaces.Discrete(3)  # 3个离散动作 (0买入, 1卖出, 2持仓)

        # 观测空间: (窗口*价格特征 + 份额 + 当前账户价值)
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(STATE_DIM,),  # OHLCV + shares + capital
            dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """
        重置环境状态
        :return: 初始观测值
        """
        self.current_step = self.window_size - 1
        self.initial_capital = self.initial_capital
        self.capital = self.initial_capital
        self.cash = self.initial_capital
        self.market_value = 0
        self.shares = 0
        self.current_date = self.df.loc[self.current_step, 'Date']
        self.history = []
        return self._next_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行动作
        :param action: 交易动作 (0,1,2) 3个离散动作 (0买入, 1卖出, 2持仓)
        :return: (observation, reward, done, info)
        """
        truncated = False
        # 检查是否结束
        if self.current_step >= len(self.df) - 1:
            return self._next_observation(), 0, True, truncated, {}

        # 获取当前价格
        current_price = self._get_current_price()

        # 执行交易动作
        self._take_action(action, current_price)

        # 更新状态
        self.current_step += 1
        self.current_date = self.df.loc[self.current_step, 'Date']
        self.capital = self.cash + self.shares * self._get_current_price()

        # 计算奖励 (可修改为更复杂的奖励函数)
        reward = self._calculate_reward()

        # 检查是否终止
        done = self.current_step > len(self.df) - 1 or self.capital <= 0

        # 记录交易信息
        info = {
            'date_trans': 'from %s to %s' % (self.df.loc[self.current_step-1, 'Date'], self.df.loc[self.current_step, 'Date']),
            'step_trans': 'from %d to %d' % (self.current_step-1, self.current_step),
            'now_capital': self.capital,
            'now_cash': self.cash,
            'now_shares': self.shares,
            'last_price': self.df.loc[self.current_step-1, 'Close'],
            'now_price': self._get_current_price()
        }
        self.history.append(info)

        return self._next_observation(), reward, done, truncated, info

    def _take_action(self, action: int, current_price: float) -> None:
        """ 执行具体交易动作 """
        if action == 0:  # 买入
            max_shares = self.cash // (current_price * (1 + self.transaction_fee))
            if max_shares > 0:
                self.shares = max_shares + self.shares
                self.market_value = self.shares * current_price
                self.cash = self.cash - max_shares * current_price - max_shares * current_price * self.transaction_fee
                self.capital = self.cash + self.market_value

        elif action == 1:  # 卖出
            if self.shares > 0:
                self.cash = self.shares * current_price * (1- self.transaction_fee) + self.cash
                self.shares = 0
                self.market_value = 0
                self.capital = self.cash

    def _calculate_reward(self) -> float:
        """ 计算奖励值 """
        # 基础奖励：净值变化
        reward = (self.capital - self.initial_capital) / self.initial_capital * 100  # 百分比变化

        # 风险惩罚（可选）
        if self.shares > 0:
            price_window = self.df['Close'][
                           self.current_step + 1 - self.window_size:self.current_step+1
                           ]
            volatility = price_window.pct_change().std()
            reward -= 0.1 * volatility * 100  # 波动率惩罚

        return reward

    def _get_current_price(self) -> float:
        """ 获取当前时间步的收盘价 """
        return self.df.loc[self.current_step, 'Close']

    def _next_observation(self) -> np.ndarray:
        """ 构建观测值 """
        # 获取价格窗口数据 (OHLCV)
        obv_bars = self.df.iloc[self.current_step + 1
                                - self.window_size:self.current_step+1][['Open', 'High', 'Low', 'Close', 'Volume']]
        price_window = obv_bars.values.flatten()

        # 标准化处理价格系列 z-score
        mean_price = obv_bars['Close'].mean()
        std_price = obv_bars['Close'].std()
        norm_price = (price_window - mean_price) / std_price
        norm_shares = 1 if self.shares > 0 else 0       # 满仓/空仓策略
        norm_capital = self.capital / self.initial_capital

        # 组合观测值
        obs = np.concatenate((
            norm_price,
            np.array([norm_shares, norm_capital])
        )).astype(np.float32)

        return obs

    def render(self, mode: str = 'human') -> None:
        """ 可视化 """
        if mode == 'human':
            capitals = [x['now_capital'] for x in self.history]
            return capitals


# 数据样本池
data_set = []
model = None
next_model = None
data_path = 'data/510300.SH_20_22.csv'
model_path = './save/DQN_simple_trading'

def get_action(state):
    # 通过神经网络, 得到一个动作
    state = torch.FloatTensor(state).reshape(1, STATE_DIM)
    return model(state).argmax().item()

# 更新样本池，向样本池中添加N条数据, 删除M条最古老的数据
def update_data(_env):
    # 初始化游戏
    state = _env.reset()

    # 玩到游戏结束为止
    done = False
    while not done:
        # 根据当前状态得到一个动作
        action = get_action(state)

        # 执行动作,得到反馈
        next_state, reward, done, _, _ = _env.step(action)

        # 记录数据样本
        data_set.append((state, action, reward, next_state, done))

        # 更新游戏状态,开始下一个动作
        state = next_state

    # 数据上限,超出时从最古老的开始删除
    while len(data_set) > TIME_RANGE:
        data_set.pop(0)


# 获取一批数据样本
def get_sample():
    # 从样本池中采样64条数据
    samples = random.sample(data_set, 64)

    # [b, STATE_DIM] state
    state = torch.FloatTensor([i[0] for i in samples]).reshape(-1, STATE_DIM)
    # [b, 1] action
    action = torch.LongTensor([i[1] for i in samples]).reshape(-1, 1)
    # [b, 1] reword
    reward = torch.FloatTensor([i[2] for i in samples]).reshape(-1, 1)
    # [b, STATE_DIM] next_state
    next_state = torch.FloatTensor([i[3] for i in samples]).reshape(-1, STATE_DIM)
    # [b, 1] done
    done = torch.LongTensor([int(i[4]) for i in samples]).reshape(-1, 1)

    return state, action, reward, next_state, done


def get_value(state, action):
    # 使用状态计算出动作的logits
    # [b, STATE_DIM] -> [b, 2]
    value = model(state)

    # 根据实际使用的action取出每一个值
    # 这个值就是模型评估的在该状态下,执行动作的分数
    # 在执行动作前,显然并不知道会得到的反馈和next_state
    # 所以这里不能也不需要考虑next_state和reward
    # [b, STATE_DIM] -> [b, 3]
    value = value.gather(dim=1, index=action)

    return value


def get_target(reward, next_state, done):
    # 上面已经把模型认为的状态下执行动作的分数给评估出来了
    # 下面使用next_state和reward计算真实的分数
    # 针对一个状态,它到底应该多少分,可以使用以往模型积累的经验评估
    # 这也是没办法的办法,因为显然没有精确解,这里使用延迟更新的next_model评估

    # 使用next_state计算下一个状态的分数
    # [b, STATE_DIM] -> [b, 3]
    with torch.no_grad():
        target = next_model(next_state)

    # 取所有动作中分数最大的
    # [b, STATE_DIM] -> [b, 3]
    target = target.max(dim=1)[0]
    target = target.reshape(-1, 1)

    # 下一个状态的分数乘以一个系数,相当于权重
    target *= 0.98

    # 如果next_state已经游戏结束,则next_state的分数是0
    # 因为如果下一步已经游戏结束,显然不需要再继续玩下去,也就不需要考虑next_state了.
    # [b, 1] * [b, 1] -> [b, 1]
    target *= (1 - done)

    # 加上reward就是最终的分数
    # [b, 1] + [b, 1] -> [b, 1]
    target += reward

    return target

def evaluate(_env):
    state = _env.reset()
    reward_sum = 0
    done = False
    while not done:
        action = get_action(state)
        state, reward, done, _, _ = _env.step(action)
        reward_sum += reward
    return reward_sum

def train(_env):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = torch.nn.MSELoss()

    # 训练N次
    for epoch in range(EPOCH_LOOP):
        # 更新N条数据
        update_data(_env)

        # 每次更新过数据后,学习N次
        for i in range(200):
            # 采样一批数据
            state, action, reward, next_state, over = get_sample()

            # 计算一批样本的value和target
            value = get_value(state, action)
            target = get_target(reward, next_state, over)

            # 更新参数
            loss = loss_fn(value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 把model的参数复制给next_model
            if (i + 1) % 10 == 0:
                next_model.load_state_dict(model.state_dict())

        if epoch % 50 == 0:
            print(epoch, len(data_set), sum([evaluate(_env) for _ in range(5)]) / 5)

    torch.save(model, model_path)

def load_model():
    global model
    # 创建模型实例
    model = torch.load('./save/DQN_simple_trading', weights_only=False)
    model.eval()

def run_trading(_env):
    state = _env.reset()
    reword_sum = 0
    done = False

    capitals = _env.render()
    while not done:
        action = get_action(state)
        next_state, reward, done, _, _ = _env.step(action)
        # print("当前state=", next_state)
        state = next_state
        capitals = _env.render()
        reword_sum += reward

        if _env.current_step >= 608:
            pass

    return reword_sum, capitals

def show_results(_df, _capitals):
    # 绘制价格曲线和净值曲线
    plt.figure(figsize=(12, 6))

    # 价格曲线
    hs300 = _df['Close'] / _df['Close'][0]
    nvs = [1 for _ in range(0, WIN_SIZE)] + list(_capitals / _capitals[0])
    plt.subplot(1, 1, 1)
    plt.plot(hs300, label='300', color='green')
    plt.plot(nvs, label='Net Value', color='red')
    plt.xticks(rotation=45)  # 旋转日期标签以便更好地显示
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv(data_path)
    env = SimpleTradingEnv(df=df, transaction_fee=0)

    # 训练模型
    model = SimpleTradingDNN()
    next_model = SimpleTradingDNN()
    train(env)

    # 加载模型并展示
    #load_model()
    #reword_sum, capitals = run_trading(env)
    #show_results(df, capitals)

