from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import wandb


config = {
    "env_name": "LunarLander-v3",
    "num_envs": 8,
    "total_steps": 500_000,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.999,
    "gae_lambda": 0.98,
    "ent_coef": 0.01,
}

train_env = make_vec_env(config["env_name"], n_envs=config["num_envs"])
run = wandb.init(
    project=config["env_name"],
    config=config,
    sync_tensorboard=True,
)

model = PPO(
    "MlpPolicy",
    train_env,
    n_steps=config["n_steps"],
    batch_size=config["batch_size"],
    n_epochs=config["n_epochs"],
    gamma=config["gamma"],
    gae_lambda=config["gae_lambda"],
    ent_coef=config["ent_coef"],
    verbose=1,
    device="cuda",
    tensorboard_log=f"runs/{run.id}",
)
# model = PPO.load("checkpoints/lunar_lander_model", train_env, device="cuda")

model.learn(total_timesteps=config["total_steps"])
model.save("checkpoints/lunar_lander_model")
train_env.close()
wandb.finish()
