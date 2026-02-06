import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import os


# --- 配置 ---
MODEL_PATH = "checkpoints/lunar_lander_model"  # 不需要加 .zip，SB3 会自动识别
VIDEO_FOLDER = "videos"
ENV_NAME = "LunarLander-v3"

# 1. 创建环境 (必须指定 render_mode='rgb_array' 才能被录制)
env = gym.make(ENV_NAME, render_mode="rgb_array")

# 2. 包装环境以录制视频
# name_prefix: 视频文件名前缀
# episode_trigger: lambda x: True 表示每一局都录
env = RecordVideo(
    env,
    video_folder=VIDEO_FOLDER,
    episode_trigger=lambda episode_id: True,
    name_prefix="ppo-lunar"
)

# 3. 加载 SB3 模型
# 这里的 device="cuda" 是指用 GPU 做推理，虽然录视频通常 CPU 也够用
print(f"Loading model from {MODEL_PATH}...")
model = PPO.load(MODEL_PATH, device="cuda")

print("Start recording...", flush=True)

# 跑 10 个 episode
total_steps = 0

for ep in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        # --- 核心修改点 ---
        # SB3 的 predict 自动处理了：
        # 1. numpy -> tensor
        # 2. 放入 cuda
        # 3. 网络前向传播
        # 4. 采样 (deterministic=True 意味着取概率最大的动作，不随机)
        action, _states = model.predict(obs, deterministic=True)

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"Episode {ep + 1}: Score = {total_reward:.2f} Steps = {steps}")
    total_steps += steps

# 关闭环境，确保视频缓冲区写入文件
env.close()

print(f"Videos saved in folder: {os.path.abspath(VIDEO_FOLDER)}")
print(f"Total steps: {total_steps}")
