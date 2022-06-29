from environment_generate import barobotGympickEnv

env = barobotGympickEnv()
obs = env.reset()

n_steps = 100
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    print(action)
    obs, reward, done, info = env.step(action)
    print(obs,reward,done,info)
    env.render()
    if done:
        obs = env.reset()