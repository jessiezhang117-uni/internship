import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env 


from gym import utils
from  environment.braccio_arm_gym import barobotGymEnv




env = barobotGymEnv(
            has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            reward_type="sparse",renders=False)


check_env(env)

# model = DQN("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("dqn_barobot")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_barobot")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     #env.render()
#     if done:
#       obs = env.reset()