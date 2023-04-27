#import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import A2C, PPO
import os

#num_cpu = 512

models_dir = "./models/LL/A2C-mlp"
logdir = "./logs/LL"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = env = make_atari_env("AssaultNoFrameskip-v4", seed=0)
env.reset()

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log = logdir)
#learn for 10k steps

time_steps = 10000
for i in range(1, 30):
    model.learn(total_timesteps=time_steps, reset_num_timesteps = False, tb_log_name="A2C-mlp")
    model.save(f"{models_dir}/{time_steps*i}")



env.close()