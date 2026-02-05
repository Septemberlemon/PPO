import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.utils.data import BatchSampler, SubsetRandomSampler
import gymnasium as gym
import wandb

from models import Actor, Critic


test_name = "PPO"
gamma = 0.99
actor_learning_rate = 0.0005
critic_learning_rate = 0.001
iterations = 100
batch_size = 4000
k_epochs = 10

env = gym.make("LunarLander-v3")

actor = Actor(env.observation_space.shape[0], env.action_space.n).to("cuda")
critic = Critic(env.observation_space.shape[0]).to("cuda")

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)

wandb.init(project="LunarLander-v3", name=test_name)


def sample():
    trajectories = []
    total_steps = 0
    while total_steps < batch_size:
        obs, info = env.reset()
        observations = [obs]
        log_probs = []
        rewards = []
        actions = []

        with torch.no_grad():
            while True:
                obs_tensor = torch.from_numpy(obs).to("cuda")
                logits = actor(obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                obs, reward, terminated, truncated, info = env.step(action.item())
                observations.append(obs)
                log_probs.append(dist.log_prob(action))
                actions.append(action)
                rewards.append(reward)

                total_steps += 1

                if terminated or truncated:
                    break

            observations = torch.from_numpy(np.stack(observations)).to("cuda")
            log_probs = torch.stack(log_probs)
            actions = torch.stack(actions)
            rewards = torch.tensor(rewards, dtype=torch.float32).to("cuda")

            GAEs = []
            returns = []
            GAE = torch.zeros([], device="cuda")
            V_s = critic(observations)
            for i in range(actions.shape[0] - 1, -1, -1):
                td_target = rewards[i].clone()
                if i != actions.shape[0] - 1 or truncated:
                    td_target += gamma * V_s[i + 1]
                GAE *= 0.95 * gamma
                GAE += td_target
                returns.append(GAE.clone())
                GAE -= V_s[i]
                GAEs.append(GAE.clone())
            GAEs.reverse()
            GAEs = torch.stack(GAEs)
            returns.reverse()
            returns = torch.stack(returns)
        observations = observations[:-1]
        trajectories.append((observations, log_probs, actions, rewards, returns, GAEs))
        print(f"steps: {actions.shape[0]}")
        print(f"reward: {rewards.sum().item()}")
        wandb.log({
            "steps": actions.shape[0],
            "reward": rewards.sum().item(),
        })
    return trajectories


# def learn(trajectories):
#     observations, log_probs, actions, rewards, returns, GAEs = (torch.cat(data) for data in zip(*trajectories))
#     for epoch in range(k_epochs):
#         critic_loss = F.mse_loss(critic(observations), returns)
#         logits = actor(observations)
#         dist = Categorical(logits=logits)
#         ratio = torch.exp(dist.log_prob(actions) - log_probs)
#         actor_loss = - torch.mean(torch.min(ratio * GAEs, torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * GAEs))
#
#         print(f"critic loss: {critic_loss.item()}")
#         print(f"actor loss: {actor_loss.item()}")
#         wandb.log({
#             "critic loss": critic_loss.item(),
#             "actor loss": actor_loss.item(),
#         })
#         critic_optimizer.zero_grad()
#         actor_optimizer.zero_grad()
#         critic_loss.backward()
#         actor_loss.backward()
#         critic_optimizer.step()
#         actor_optimizer.step()

def learn(trajectories):
    # 1. 整理数据
    observations, log_probs, actions, rewards, returns, GAEs = (torch.cat(data) for data in zip(*trajectories))

    # --- 【关键点 1】 Advantage Normalization (优势归一化) ---
    # 这行代码是 PPO 收敛的核心，防止梯度爆炸或消失
    GAEs = (GAEs - GAEs.mean()) / (GAEs.std() + 1e-8)

    # 准备 Mini-batch 参数
    batch_size_total = observations.size(0)
    mini_batch_size = 64  # 推荐 64 或 128

    for epoch in range(k_epochs):
        # --- 【关键点 3】 Mini-batch Training (打乱数据分批训练) ---
        # 使用随机采样器生成索引
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size_total)), mini_batch_size, drop_last=False)

        for indices in sampler:
            # 取出当前 mini-batch 的数据
            mb_obs = observations[indices]
            mb_actions = actions[indices]
            mb_log_probs = log_probs[indices]
            mb_returns = returns[indices]
            mb_GAEs = GAEs[indices]

            # 前向传播
            new_logits = actor(mb_obs)
            new_values = critic(mb_obs).squeeze()  # 确保维度匹配
            dist = Categorical(logits=new_logits)

            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()  # 计算熵

            # 计算 Ratio
            ratio = torch.exp(new_log_probs - mb_log_probs)

            # Actor Loss (PPO Clip)
            surr1 = ratio * mb_GAEs
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * mb_GAEs

            # --- 【关键点 2】 Entropy Bonus (熵正则项) ---
            # 减去 entropy * 0.01，鼓励探索，防止飞船死板地悬停
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            # Critic Loss
            critic_loss = F.mse_loss(new_values, mb_returns)

            # 总 Loss (通常合并反向传播)
            total_loss = actor_loss + 0.5 * critic_loss

            # 优化更新
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()
            total_loss.backward()

            # 建议加上梯度裁剪，防止梯度突然爆炸
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)

            critic_optimizer.step()
            actor_optimizer.step()

        # 日志记录 (记录最后一个 batch 的情况即可)
        wandb.log({
            "critic loss": critic_loss.item(),
            "actor loss": actor_loss.item(),  # 注意这里的 actor loss 包含了熵
            "entropy": entropy.item()  # 监控熵也很重要，熵如果掉太快说明过拟合了
        })
        print(f"critic loss: {critic_loss.item()} | actor loss: {actor_loss.item()}")


if __name__ == "__main__":
    for iteration in range(iterations):
        print(f"Iteration: {iteration}")
        trajectories = sample()
        learn(trajectories)

    env.close()
    wandb.finish()
    torch.save(actor.state_dict(), f"checkpoints/{test_name}_actor.pth")
    torch.save(critic.state_dict(), f"checkpoints/{test_name}_critic.pth")
