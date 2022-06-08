import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
import parser
import gym
from bullet.braccio_arm_Gym import braccio_arm_gym
from stable_baselines import DQN, PPO2, DDPG
import datetime
import time
from stable_baselines.ddpg.policies import DDPGPolicy,MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import numpy as np
from stable_baselines.common.vec_env import VecVideoRecorder
import imageio

def runsimulation(model,env,iterations):
    obs = env.reset()
    time_step_counter = 0
    iterations = iterations
    while time_step_counter < iterations:
        action, _ = model.predict(obs)
        obs, rewards, dones, _ = env.step(action)
        time_step_counter +=1

        if dones:
            obs = env.reset()

def evaluate(model,num_episodes=100):
    '''
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL agent
    :param num_episodes: (int) number of episodes to evaluate
    :return: (float) mean reward for the last num_episodes
    '''
    env = model.get_env()
    all_episode_rewards =[]
    for i in range(num_episodes):
        episode_rewards=[]
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))
    
    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:",mean_episode_reward,"Num episodes:",num_episodes)

def record_gif(model,env,name):
    images=[]
    obs = model.env.reset()
    img = model.env.render()
    for i in range(500):
        images.append(img)
        action,_ = model.predict(obs)
        obs,_,_,_ = model.env.step(action)
        img = model.env.render()
    imageio.mimsave(name,[np.array(img) for i,img in enumerate(images) if i%2 ==0],fps=29)

def record_video(env_id,model,video_length=500,prefix='',video_folder='videos/'):
    eval_env = DummyVecEnv([lambda:gym.make(env_id)])
    eval_env = VecVideoRecorder(env,video_folder=video_folder,record_video_trigger=lambda step: step==0,video_length=video_length,name_prefix=prefix)
    obs = eval_env.reset()
    for _ in range(video_length):
        action,_ = model.predict(obs)
        obs,_,_,_ = eval_env.step(action)

    eval_env.close()

def savemodel(model,MODEL,ENVIRONMENT,DATE):
    name = "trainedmodel_%s_%s_%s.pkl" %(MODEL,ENVIRONMENT,DATE)
    print("save model as:",name)
    model.save(name)

class braccioWrapper(gym.Wrapper):
    def _init_(self,env):
        #Call the parent constructor, se we can access self.env later
        super(braccioWrapper,self)._init_(env)

    def reset(self):
        obs = self.env.reset()
        return obs
    
    def step(self,action):
        obs,reward,done,info = self.env.step(action)
        return obs,reward,done,info

if __name__ =="__main__":
    ################################## Filter tensorflow version warnings
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings

    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf

    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging

    tf.get_logger().setLevel(logging.ERROR)

    args=parser.arg_parse()

    # Create save dir
    model_dir = args.model_dir

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    ############ load environment
    env = braccio_arm_gym(renders=False,isDiscrete=True)
    env = DummyVecEnv([lambda:env]) # vectorized environments

    ############ MODELS
    model = PPO2('MlpPolicy', 'Pendulum-v0', verbose=0).learn(8000)
    # The model will be saved under PPO2_tutorial.zip
    #ddpg_model = DDPG(MlpPolicy, env, verbose=1, param_noise=None, random_exploration=0.1)
    kwargs = {'double_q': True, 'prioritized_replay': True, 'policy_kwargs': dict(dueling=True)}  #DQN + Prioritized Experience Replay + Double Q-Learning + Dueling
    dqn_model = DQN('MlpPolicy', env, verbose=1, **kwargs)

    ########### learn & evaluate
    mean_reward_before_train = evaluate(dqn_model,num_episodes=50)
    dqn_model.learn(10000)
    mean_reward_after_train = evaluate(dqn_model,num_episodes=50)

    ########### save model
    model.save(args.model_dir)


