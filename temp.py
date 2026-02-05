import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import gymnasium as gym
import wandb
import os

# 假设 models.py 内容如下 (如果没有该文件，请取消注释以下类定义)
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================

test_name = "PPO_Fixed"
gamma = 0.99
gae_lambda = 0.95
actor_learning_rate = 3e-4  # 稍微降低一点LR更稳定
critic_learning_rate = 1e-3
iterations = 500  # 增加迭代次数
batch_size = 2048  # PPO通常按步数采集，而不是按"几条轨迹"
k_epochs = 10
entropy_coef = 0.01  # 增加熵正则系数
clip_param = 0.2

env = gym.make("LunarLander-v3")
device = "cuda" if torch.cuda.is_available() else "cpu"

actor = Actor(env.observation_space.shape[0], env.action_space.n).to(device)
critic = Critic(env.observation_space.shape[0]).to(device)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)


# 如果你没有wandb账号，可以注释掉这行
wandb.init(project="LunarLander-v3", name=test_name)

def sample():
    # 存储所有步的数据
    batch_obs = []
    batch_actions = []
    batch_log_probs = []
    batch_rewards = []
    batch_dones = []
    batch_vals = []

    # 计数器
    step_count = 0
    total_rewards = []  # 记录每一局的总分用于打印

    while step_count < batch_size:
        obs, info = env.reset()
        episode_reward = 0

        # 临时存储当前轨迹
        traj_obs = []
        traj_actions = []
        traj_log_probs = []
        traj_rewards = []
        traj_dones = []
        traj_vals = []

        while True:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                logits = actor(obs_tensor)
                val = critic(obs_tensor)

                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            traj_obs.append(obs)
            traj_actions.append(action.item())
            traj_log_probs.append(log_prob.item())
            traj_rewards.append(reward)
            traj_dones.append(done)
            traj_vals.append(val.item())

            episode_reward += reward
            obs = next_obs
            step_count += 1

            if done:
                # 处理最后一步的 Value，用于计算 GAE
                with torch.no_grad():
                    next_val = 0
                    if truncated:  # 如果是超时截断，需要用网络预估下一个价值
                        next_obs_tensor = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
                        next_val = critic(next_obs_tensor).item()

                # 计算 GAE 和 Returns
                traj_advantages = []
                gae = 0
                for t in range(len(traj_rewards) - 1, -1, -1):
                    # 如果是最后一步，使用 next_val，否则使用下一时刻的 V
                    if t == len(traj_rewards) - 1:
                        next_v = next_val
                        next_non_terminal = 1.0 - (1.0 if terminated else 0.0)  # truncated不算terminal
                    else:
                        next_v = traj_vals[t + 1]
                        next_non_terminal = 1.0  # 轨迹中间肯定没结束

                    delta = traj_rewards[t] + gamma * next_v * next_non_terminal - traj_vals[t]
                    gae = delta + gamma * gae_lambda * next_non_terminal * gae
                    traj_advantages.insert(0, gae)

                # 将轨迹数据加入 Batch
                batch_obs.extend(traj_obs)
                batch_actions.extend(traj_actions)
                batch_log_probs.extend(traj_log_probs)
                # PPO Critic Target: Returns = Advantage + Value
                batch_returns = [adv + val for adv, val in zip(traj_advantages, traj_vals)]
                batch_rewards.extend(batch_returns)  # 注意：这里存的是Target Value (Returns)

                total_rewards.append(episode_reward)
                break

            if step_count >= batch_size:
                # 如果还没做完这一局就超出了步数限制，PPO通常有两种做法：
                # 1. 丢弃这半局 2. 强制截断。为了代码简单，我们允许它跑完这局稍微超出一点batch_size
                pass

    # 转换为 Tensor
    batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32).to(device)
    batch_actions = torch.tensor(batch_actions, dtype=torch.int64).to(device)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32).to(device)
    batch_returns = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
    batch_vals = torch.tensor(batch_vals, dtype=torch.float32).to(device)

    # 重新计算 Advantages 用于 Actor Loss (避免 detach 问题)
    batch_advantages = batch_returns - batch_vals

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Collected {len(batch_obs)} steps, Avg Reward: {avg_reward:.2f}")
    wandb.log({"avg_reward": avg_reward})

    return batch_obs, batch_actions, batch_log_probs, batch_returns, batch_advantages


def learn(obs, actions, old_log_probs, returns, advantages):
    # 关键点：Advantage Normalization
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = obs.size(0)

    for _ in range(k_epochs):
        # 随机打乱数据 (Full batch shuffle)
        indices = torch.randperm(dataset_size)

        # 为了简单不写 Mini-batch loop，但建议实际使用时切分 Mini-batch (例如64或128)
        # 这里直接全量 update 也是可以跑通 LunarLander 的

        # 重新评估新的 Logits 和 Values
        logits = actor(obs[indices])
        new_values = critic(obs[indices]).squeeze()

        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions[indices])
        dist_entropy = dist.entropy()

        # 计算 Ratio
        ratio = torch.exp(new_log_probs - old_log_probs[indices])

        # Actor Loss (Clipped)
        surr1 = ratio * advantages[indices]
        surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages[indices]
        actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * dist_entropy.mean()

        # Critic Loss (MSE against Returns)
        critic_loss = F.mse_loss(new_values, returns[indices])

        # Update
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()

    print(f"Loss - Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f}")


if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    for i in range(iterations):
        print(f"\nIteration {i + 1}/{iterations}")
        obs, actions, log_probs, returns, advantages = sample()
        learn(obs, actions, log_probs, returns, advantages)

        if (i + 1) % 20 == 0:
            torch.save(actor.state_dict(), f"checkpoints/{test_name}_actor.pth")
            torch.save(critic.state_dict(), f"checkpoints/{test_name}_critic.pth")
