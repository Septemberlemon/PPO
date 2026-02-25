import pickle

import gymnasium as gym
import torch
from torch.distributions import Categorical, Normal

from models import Actor  # 确保 model.py 在同一目录下


# 配置
env_name = "Ant-v5"
env = gym.make(env_name)
model_path = f"checkpoints/{env_name}/PPO_actor.pth"  # 训练保存的路径
running_mean_std_path = f"checkpoints/{env_name}/PPO_rms.pkl"
video_folder = f"videos/{env_name}"  # 视频保存目录
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 创建环境，必须指定 render_mode='rgb_array' 才能录像
env = gym.make(env_name, render_mode='rgb_array')

# 2. 包装环境以录制视频
# episode_trigger 决定录制哪些局，这里 lambda x: True 表示每一局都录
env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)

# 3. 加载模型
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
actor = Actor(state_dim, action_dim, [256, 256]).to(device)
actor.load_state_dict(torch.load(model_path))

with open(running_mean_std_path, "rb") as f:
    running_mean_std = pickle.load(f)

print("Start recording...", flush=True)

# 跑 10 个 episode 看看效果
total_steps = 0
for ep in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        # 变成 Tensor
        obs_tensor = torch.tensor(running_mean_std.normalize(obs, update=False), dtype=torch.float32).to(device)
        mu, std = actor(obs_tensor)
        dist = Normal(mu, std)
        action = dist.sample()
        obs, reward, terminated, truncated, info = env.step(torch.clamp(action, -1, 1).cpu().numpy())
        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"Episode {ep + 1}: Score = {total_reward} Steps = {steps}")
    total_steps += steps

env.close()
print(f"Videos saved in folder: {video_folder}")
print(f"Total steps: {total_steps}")
