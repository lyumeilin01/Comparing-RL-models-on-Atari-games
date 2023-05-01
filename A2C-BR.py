#import gymnasium as gym
from collections import OrderedDict

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import A2C, PPO, DQN
import os

from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import VecFrameStack

models_dir = "./models/BR/A2C"
logdir = "./logs/BR"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = make_atari_env("BreakoutNoFrameskip-v4", seed=0, n_envs=16)
env = VecFrameStack(env, 4)
env.reset()

hparams = OrderedDict([('ent_coef', 0.01),
             ('policy_kwargs',
              dict(optimizer_class=RMSpropTFLike,
              optimizer_kwargs=dict(eps=1e-5))),
             ('vf_coef', 0.25)])



model = A2C("CnnPolicy", env, verbose=1, tensorboard_log = logdir, **hparams)
#learn for 10k steps

time_steps = 100000
for i in range(1, 30):
    model.learn(total_timesteps=time_steps, reset_num_timesteps = False, tb_log_name="A2C")
    if i % 5 == 0:  # every 50k save
        model.save(f"{models_dir}/{time_steps*i}")


env.close()