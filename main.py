import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.utils.data import BatchSampler, SubsetRandomSampler
import gymnasium as gym
import wandb

from models import Actor, Critic


env_name = "LunarLander-v3"
test_name = "PPO"
gamma = 0.99
gae_lambda = 0.95
actor_learning_rate = 0.0005
critic_learning_rate = 0.001
iterations = 100
n_steps = 4000
n_epochs = 10
batch_size = 128
clip_range = 0.2

env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
actor = Actor(state_dim, action_dim).to("cuda")
critic = Critic(state_dim).to("cuda")

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)

wandb.init(project=env_name, name=test_name)


def rollout():
    trajectories = []
    total_steps = 0
    while total_steps < n_steps:
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

            gaes = []
            returns = []
            gae = torch.zeros([], device="cuda")
            values = critic(observations)
            trajectory_length = actions.shape[0]
            for i in range(trajectory_length - 1, -1, -1):
                td_target = rewards[i].clone()
                if i != trajectory_length - 1 or truncated:
                    td_target += gamma * values[i + 1]
                gae *= gae_lambda * gamma
                gae += td_target
                returns.append(gae.clone())
                gae = gae - values[i]
                gaes.append(gae.clone())
            gaes.reverse()
            gaes = torch.stack(gaes)
            returns.reverse()
            returns = torch.stack(returns)
        observations = observations[:-1]
        trajectories.append((observations, log_probs, actions, returns, gaes))
        print(f"steps: {actions.shape[0]}")
        print(f"reward: {rewards.sum().item()}")
        wandb.log({
            "steps": actions.shape[0],
            "reward": rewards.sum().item(),
        })
    return trajectories


def learn(trajectories, entropy_coef):
    observations, log_probs, actions, returns, gaes = (torch.cat(data) for data in zip(*trajectories))
    gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    for epoch in range(n_epochs):
        sampler = BatchSampler(SubsetRandomSampler(range(actions.shape[0])), batch_size, False)
        total_critic_loss = 0
        total_actor_loss = 0
        for indices in sampler:
            mb_observations = observations[indices]
            mb_log_probs = log_probs[indices]
            mb_actions = actions[indices]
            mb_returns = returns[indices]
            mb_gaes = gaes[indices]

            critic_loss = F.mse_loss(critic(mb_observations), mb_returns)

            logits = actor(mb_observations)
            dist = Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(mb_actions) - mb_log_probs)
            actor_loss = - torch.mean(torch.min(ratio * mb_gaes, torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * mb_gaes))

            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()

            actor_loss -= entropy_coef * dist.entropy().mean()

            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
            critic_optimizer.step()
            actor_optimizer.step()

        print(f"critic loss: {total_critic_loss} | actor loss: {total_actor_loss}")
        wandb.log({
            "critic loss": total_critic_loss,
            "actor loss": total_actor_loss,
        })


if __name__ == "__main__":
    for iteration in range(iterations):
        print(f"Iteration: {iteration}")
        trajectories = rollout()
        learn(trajectories, entropy_coef=0.01 * (1 - iteration / iterations) * 0)

    env.close()
    wandb.finish()
    torch.save(actor.state_dict(), f"checkpoints/{test_name}_actor.pth")
    torch.save(critic.state_dict(), f"checkpoints/{test_name}_critic.pth")
