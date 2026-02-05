import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import gymnasium as gym
import wandb
from torch.utils.data import BatchSampler, SubsetRandomSampler

# 假设你的 models.py 保持不变，或者使用简单的 MLP
# from models import Actor, Critic
# 为了代码可运行，我在这里补上简单的 Actor/Critic 定义
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # Critic 输出通常是 (batch,) 而不是 (batch, 1)


# --- Hyperparameters ---
test_name = "PPO_Fixed"
gamma = 0.99
lam = 0.95  # GAE lambda
actor_lr = 3e-4  # 标准 PPO 学习率
critic_lr = 1e-3
max_training_timesteps = 1_000_000  # 总训练步数
update_timestep = 2048  # 每次更新收集的步数 (batch_size)
k_epochs = 10  # 每次更新循环多少轮
mini_batch_size = 64  # 小批次大小
clip_coef = 0.2  # PPO Clip 范围
ent_coef = 0.01  # 熵系数，防止过早收敛
max_grad_norm = 0.5  # 梯度裁剪

device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("LunarLander-v3")
# env = gym.make("LunarLander-v3", render_mode="human") # 如果想看画面

actor = Actor(env.observation_space.shape[0], env.action_space.n).to(device)
critic = Critic(env.observation_space.shape[0]).to(device)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)


# wandb.init(project="LunarLander-v3", name=test_name) # 调试时可注释

class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []


buffer = PPOBuffer()


def sample_step(obs):
    """采集单步数据"""
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        logits = actor(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = critic(obs_tensor)

    next_obs, reward, terminated, truncated, info = env.step(action.item())
    done = terminated or truncated

    # 存入 Buffer
    buffer.states.append(torch.FloatTensor(obs))
    buffer.actions.append(action)
    buffer.log_probs.append(log_prob)
    buffer.rewards.append(reward)
    buffer.dones.append(done)
    buffer.values.append(value)

    return next_obs, done


def update():
    # 1. 整理数据为 Tensor
    states = torch.stack(buffer.states).to(device)
    actions = torch.stack(buffer.actions).squeeze().to(device)
    old_log_probs = torch.stack(buffer.log_probs).squeeze().to(device)
    rewards = buffer.rewards
    dones = buffer.dones
    values = torch.stack(buffer.values).squeeze().to(device)

    # 2. 计算 GAE (Generalized Advantage Estimation)
    advantages = []
    returns = []
    gae = 0

    # 需要下一个状态的价值来计算最后一个步的 delta
    with torch.no_grad():
        # 这里简化处理：假设最后一个状态后接的价值是0 (如果done) 或者 critic预测值
        # 为严谨起见，PPO通常会在循环结束后多采一步 next_value，这里简化为0或沿用
        next_value = 0

        # 逆序计算
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_non_terminal = 1.0 - float(dones[i])
            next_val = next_value  # 简化
        else:
            next_non_terminal = 1.0 - float(dones[i])
            next_val = values[i + 1]

        delta = rewards[i] + gamma * next_val * next_non_terminal - values[i]
        gae = delta + gamma * lam * next_non_terminal * gae
        # 下面这一步修正了你代码中的核心错误
        # Return = Advantage + Value
        ret = gae + values[i]

        advantages.insert(0, gae)
        returns.insert(0, ret)

    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    # 3. 关键：Advantage Normalization
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 4. Mini-batch 更新
    dataset_size = len(states)
    for _ in range(k_epochs):
        sampler = BatchSampler(SubsetRandomSampler(range(dataset_size)), mini_batch_size, drop_last=False)

        for indices in sampler:
            mb_states = states[indices]
            mb_actions = actions[indices]
            mb_old_log_probs = old_log_probs[indices]
            mb_advantages = advantages[indices]
            mb_returns = returns[indices]

            # 重新评估 New Policy
            logits = actor(mb_states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_actions)
            dist_entropy = dist.entropy()
            new_values = critic(mb_states)

            # Ratio
            ratio = torch.exp(new_log_probs - mb_old_log_probs)

            # Actor Loss (Clipped)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - ent_coef * dist_entropy.mean()

            # Critic Loss (MSE)
            critic_loss = F.mse_loss(new_values, mb_returns)

            # Update
            optimizer_total_loss = actor_loss + 0.5 * critic_loss

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            optimizer_total_loss.backward()

            # 梯度裁剪 (可选，但推荐)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)

            actor_optimizer.step()
            critic_optimizer.step()

    # 计算一些统计量用于打印
    return returns.mean().item(), actor_loss.item(), critic_loss.item()


if __name__ == "__main__":
    obs, info = env.reset()
    total_timesteps = 0
    iteration = 0

    while total_timesteps < max_training_timesteps:
        buffer.clear()

        # 1. 采集数据 (按步数采集，而不是按 episode)
        for _ in range(update_timestep):
            obs, done = sample_step(obs)
            total_timesteps += 1
            if done:
                obs, info = env.reset()

        # 2. 学习
        avg_return, a_loss, c_loss = update()
        iteration += 1

        # 打印日志
        print(f"Iter: {iteration} | Steps: {total_timesteps} | Avg Return: {avg_return:.2f} | Actor Loss: {a_loss:.4f}")

        # wandb.log(...)

    env.close()
    # wandb.finish()
