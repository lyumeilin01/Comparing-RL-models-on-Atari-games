#import gymnasium as gym
from collections import OrderedDict

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import A2C, PPO, DQN
import os

from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import VecFrameStack

#num_cpu = 512

models_dir = "./models/BR/A2C"
logdir = "./logs/BR"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = make_atari_env("BreakoutNoFrameskip-v4", seed=0, n_envs=1)
env = VecFrameStack(env, 4)
env.reset()
hparams = OrderedDict([('batch_size', 32),
             ('exploration_final_eps', 0.01),
             ('exploration_fraction', 0.1),
             ('frame_stack', 4),
             ('gradient_steps', 1),
             ('learning_rate', 0.0001),
             ('train_freq', 4)])

model = DQN("CnnPolicy", env, verbose=1, tensorboard_log = logdir, **hparams)
#learn for 10k steps

time_steps = 100000
for i in range(1, 30):
    model.learn(total_timesteps=time_steps, reset_num_timesteps = False, tb_log_name="DQN")
    if i % 5 == 0:  # every 50k save
        model.save(f"{models_dir}/{time_steps*i}")


# episodes = 10
# for ep in range(episodes):
#     state = env.reset()
#     done = False
#     while not done:
#         env.render()
#         action, _ = model.predict(state)
#         state, reward, done, _, _ = env.step(action)


env.close()