from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import A2C, PPO
#KungFuMaster
#Breakout
#Assault
#Alien

env = make_atari_env("BreakoutNoFrameskip-v4", seed=0)
env.reset()
models_dir = "./models/BR/A2C"

model_path = f"{models_dir}/2500000.zip"
model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    state = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(state)
        state, reward, done, _ = env.step(action)
env.close()
