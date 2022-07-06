from braccio_gym_test import BraccioArmGymEnv
from stable_baselines3.common.env_checker import check_env 
env = BraccioArmGymEnv(renders=False, isDiscrete=False)
check_env(env)
from stable_baselines3 import DDPG
model = DDPG(policy="MlpPolicy", env=env, verbose=1)
model.learn(total_timesteps=100)
model.save("ddpg_barobot")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DDPG.load("ddpg_barobot",env=env)

from stable_baselines3.common.evaluation import evaluate_policy
# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()