import gymnasium as gym
import torch
from torch.distributions import Categorical

from models import Actor  # 确保 model.py 在同一目录下


# 配置
model_path = "checkpoints/PPO_actor.pth"  # 你训练保存的路径
video_folder = "videos"  # 视频保存目录
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 创建环境，必须指定 render_mode='rgb_array' 才能录像
env = gym.make('LunarLander-v3', render_mode='rgb_array')

# 2. 包装环境以录制视频
# episode_trigger 决定录制哪些局，这里 lambda x: True 表示每一局都录
env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)

# 3. 加载模型
actor = Actor(env.observation_space.shape[0], env.action_space.n).to(device)
actor.load_state_dict(torch.load(model_path))

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
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        logits = actor(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"Episode {ep + 1}: Score = {total_reward} Steps = {steps}")
    total_steps += steps

env.close()
print(f"Videos saved in folder: {video_folder}")
print(f"Total steps: {total_steps}")
