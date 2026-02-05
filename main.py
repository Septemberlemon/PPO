import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import gymnasium as gym
import wandb

from models import Actor, Critic


test_name = "PPO"
gamma = 0.99
actor_learning_rate = 0.0005
critic_learning_rate = 0.001
iterations = 10
importance_sampling_size = 100
k_epochs = 10

env = gym.make("LunarLander-v3")

actor = Actor(env.observation_space.shape[0], env.action_space.n).to("cuda")
critic = Critic(env.observation_space.shape[0]).to("cuda")

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)

wandb.init(project="LunarLander-v3", name=test_name)


def sample():
    trajectories = []
    for _ in range(importance_sampling_size):
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

                if terminated or truncated:
                    break

            observations = torch.from_numpy(np.stack(observations)).to("cuda")
            log_probs = torch.stack(log_probs)
            actions = torch.stack(actions)
            rewards = torch.tensor(rewards, dtype=torch.float32).to("cuda")

            GAEs = []
            GAE = torch.zeros([], device="cuda")
            V_s = critic(observations)
            for i in range(actions.shape[0] - 1, -1, -1):
                td_target = rewards[i]
                if i != actions.shape[0] - 1 or truncated:
                    td_target += gamma * V_s[i + 1]
                td_error = td_target - V_s[i]
                GAE *= 0.95 * gamma
                GAE += td_error
                GAEs.append(GAE.clone())
            GAEs.reverse()
            GAEs = torch.stack(GAEs)
        observations = observations[:-1]
        trajectories.append((observations, log_probs, actions, rewards, GAEs))
        print(f"steps: {actions.shape[0]}")
        print(f"reward: {rewards.sum().item()}")
        wandb.log({
            "steps": actions.shape[0],
            "reward": rewards.sum().item(),
        })
    return trajectories


def learn(trajectories):
    observations, log_probs, actions, rewards, GAEs = (torch.cat(data) for data in zip(*trajectories))
    for epoch in range(k_epochs):
        critic_loss = F.mse_loss(critic(observations[:-1]), rewards[:-1] + gamma * critic(observations[1:]).detach())
        logits = actor(observations)
        dist = Categorical(logits=logits)
        ratio = torch.exp(dist.log_prob(actions) - log_probs)
        actor_loss = - torch.mean(torch.min(ratio * GAEs, torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * GAEs))

        print(f"critic loss: {critic_loss.item()}")
        print(f"actor loss: {actor_loss.item()}")
        wandb.log({
            "critic loss": critic_loss.item(),
            "actor loss": actor_loss.item(),
        })
        critic_optimizer.zero_grad()
        actor_optimizer.zero_grad()
        critic_loss.backward()
        actor_loss.backward()
        critic_optimizer.step()
        actor_optimizer.step()


if __name__ == "__main__":
    for iteration in range(iterations):
        print(f"Iteration: {iteration}")
        trajectories = sample()
        learn(trajectories)

    torch.save(actor.state_dict(), f"checkpoints/{test_name}_actor.pth")
    torch.save(critic.state_dict(), f"checkpoints/{test_name}_critic.pth")
