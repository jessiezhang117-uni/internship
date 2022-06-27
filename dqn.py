from environment_generate import barobotGympickEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import time

env = barobotGympickEnv()

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=100, n_eval_episodes=100,
                             deterministic=True, render=False)

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=("tensorboard/") ,
              gamma=0.99, learning_rate=0.0005, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02,
               train_freq=1, batch_size=32, double_q=True, learning_starts=1000,
              target_network_update_freq=500, prioritized_replay=True, prioritized_replay_alpha=0.6,
              prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06,
              param_noise=False, _init_setup_model=True,
              policy_kwargs=None, full_tensorboard_log=False)

model.learn(total_timesteps=100000, callback=eval_callback, 
            tb_log_name="dqn")

model.save("dqn")
