
import gym
from environment_generate import barobotGympickEnv
from stable_baselines3 import DDPG


def main():

    env = barobotGympickEnv()

    model = DDPG.load("ddpg_barobot")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
  main()