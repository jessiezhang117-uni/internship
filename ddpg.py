from environment_generate import barobotGympickEnv

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

env = barobotGympickEnv()

eval_callback = EvalCallback(env, best_model_save_path='./ddpg/',
                             log_path='./ddpg/', eval_freq=100, n_eval_episodes=100,
                             deterministic=True, render=False)

model = DDPG("MlpPolicy", env, verbose=1, n_steps=128, batch_size=64, ent_coef=0.005, n_epochs=10, tensorboard_log="/home/morin/catkin_ws/src/hiprl_replicate/Visualization/Baselines/")
model.learn(total_timesteps=900000, callback=eval_callback, tb_log_name="ddpg")


model.save("ddpg")

