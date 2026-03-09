import pickle

import numpy as np
import torch
from torch import inference_mode
from torch.distributions import Normal
import torch.nn.functional as F
from torch.utils.data import BatchSampler, SubsetRandomSampler
import gymnasium as gym
import wandb

from models import NewActor, Critic, KvCache
from RunningMeanStd import RunningMeanStd


env_name = "Ant-v5"
test_name = "PPO"
gamma = 0.99
gae_lambda = 0.95
actor_learning_rate = 0.00005
critic_learning_rate = 0.0001
iterations = 200
n_steps = 5000
n_epochs = 10
batch_size = 128
clip_range = 0.2

env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

running_mean_std = RunningMeanStd(state_dim)

actor = NewActor(state_dim, action_dim, 64, 8).to("cuda")
critic = Critic(state_dim, [256, 256]).to("cuda")

actor.load_state_dict(torch.load(f"checkpoints/{env_name}/PPO_actor.pth"))
critic.load_state_dict(torch.load(f"checkpoints/{env_name}/PPO_critic.pth"))

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)

wandb.init(project=env_name, name=test_name)


def rollout():
    trajectories = []
    total_steps = 0
    while total_steps < n_steps:
        obs, info = env.reset()
        obs = running_mean_std.normalize(obs)
        observations = [torch.from_numpy(obs).to(torch.float32).to("cuda")]
        log_prob_densities = []
        rewards = []
        actions = []
        kv_cache = KvCache()

        with torch.no_grad():
            while True:
                if actions:
                    mu, std = actor.inference((actions[-1], observations[-1]), kv_cache)
                else:
                    mu, std = actor.inference(observations[-1], kv_cache)
                dist = Normal(mu, std)
                action = dist.sample()
                obs, reward, terminated, truncated, info = env.step(torch.clamp(action.detach(), -1, 1).cpu().numpy())

                obs = running_mean_std.normalize(obs)
                observations.append(torch.from_numpy(obs).to(torch.float32).to("cuda"))
                log_prob_densities.append(dist.log_prob(action))
                actions.append(action)
                rewards.append(reward)

                total_steps += 1

                if terminated or truncated:
                    break

            observations = torch.stack(observations)
            log_prob_densities = torch.stack(log_prob_densities)
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
        trajectories.append((observations, log_prob_densities, actions, returns, gaes))
        print(f"steps: {actions.shape[0]}")
        print(f"reward: {rewards.sum().item()}")
        wandb.log({
            "steps": actions.shape[0],
            "reward": rewards.sum().item(),
        })
    return trajectories


def learn(trajectories, entropy_coef):
    observations, log_prob_densities, actions, returns, gaes = zip(*trajectories)

    all_observations = torch.cat(observations)
    all_log_prob_densities = torch.cat(log_prob_densities)
    all_actions = torch.cat(actions)
    all_returns = torch.cat(returns)
    all_gaes = torch.cat(gaes)
    all_gaes = (all_gaes - all_gaes.mean()) / (all_gaes.std() + 1e-8)

    for epoch in range(n_epochs):
        critic_loss = F.mse_loss(critic(all_observations), all_returns)

        new_mu, new_std = actor(list(zip(observations, actions)))
        dist = Normal(new_mu, new_std)
        ratio = torch.exp((dist.log_prob(all_actions) - all_log_prob_densities).sum(dim=1))
        actor_loss = - torch.mean(
            torch.min(ratio * all_gaes, torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * all_gaes))

        total_critic_loss = critic_loss.item()
        total_actor_loss = actor_loss.item()

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
        learn(trajectories,
              entropy_coef=0.01 * (1 - iteration / iterations))

    env.close()
    wandb.finish()
    torch.save(actor.state_dict(), f"checkpoints/{env_name}/{test_name}_actor.pth")
    torch.save(critic.state_dict(), f"checkpoints/{env_name}/{test_name}_critic.pth")

    with open(f"checkpoints/{env_name}/{test_name}_rms.pkl", "wb") as f:
        pickle.dump(running_mean_std, f)
