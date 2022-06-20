from environment.braccio_arm_gym import barobotGymEnv
import numpy as np
from stable_baselines3 import DQN, DDPG
from utils.helpers import savemodel
from datetime import date
import time
import parser
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common import set_global_seeds
import matplotlib.pyplot as plt
import os

######################### PARAMETERSssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss


args = parser.arg_parse()
#set_global_seeds(args.random_seed)
start = time.time()
MODEL = args.algorithm
DISCRETE = args.discrete
DATE = date.today().strftime("%d-%m")
RENDERS = True
log_dir = ("./logdir_segmentation_%s_%s_%s/") % (MODEL,DATE)
print('Logfiles saved under:', log_dir)
os.makedirs(log_dir, exist_ok=True)

time_steps = args.timesteps
n_steps = 0

################ MODEL AND GYM ENVIRONMENT



env = barobotGymEnv(renders=RENDERS, isDiscrete=DISCRETE)
env = Monitor(env, os.path.join(log_dir, 'monitor.csv'), allow_early_resets=True)

if MODEL == 'DQN':
    from stable_baselines3.dqn import CnnPolicy, MlpPolicy



    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=(log_dir + "tensorboard_%s_%s_%s/") % (MODEL,DATE) ,
              gamma=0.99, learning_rate=0.0005, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02,
               train_freq=1, batch_size=32, double_q=True, learning_starts=1000,
              target_network_update_freq=500, prioritized_replay=True, prioritized_replay_alpha=0.6,
              prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06,
              param_noise=False, _init_setup_model=True,
              policy_kwargs=None, full_tensorboard_log=False)

if MODEL == 'DDPG':
  #from stable_baselines.ddpg import AdaptiveParamNoiseSpec
  #param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
  model = DDPG("CnnPolicy", env, verbose=1, random_exploration=0.1,tensorboard_log=(log_dir + "tensorboard_%s_%s_%s/") % (MODEL, DATE) )


################ CALLBACK FCTS

def get_callback_vars(model, **kwargs):
    """
    Helps store variables for the callback functions
    :param model: (BaseRLModel)
    :param **kwargs: initial values of the callback variables
    """
    # save the called attribute in the model
    if not hasattr(model, "_callback_vars"):
        model._callback_vars = dict(**kwargs)
    else: # check all the kwargs are in the callback variables
        for (name, val) in kwargs.items():
            if name not in model._callback_vars:
                model._callback_vars[name] = val
    return model._callback_vars # return dict reference (mutable)



def auto_save_callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    # get callback variables, with default values if unintialized
    callback_vars = get_callback_vars(_locals["self"], n_steps=0, best_mean_reward=-np.inf)
    # skip every 20 steps
    if callback_vars["n_steps"] % 20 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])

            # New best model, you could save the agent here
            if mean_reward > callback_vars["best_mean_reward"]:
                callback_vars["best_mean_reward"] = mean_reward
                # Example for saving best model
                print("Saving new best model at {} timesteps".format(x[-1]))
                _locals['self'].save(log_dir + 'best_model')
    callback_vars["n_steps"] += 1
    return True


def plotting_callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    # get callback variables, with default values if unintialized
    callback_vars = get_callback_vars(_locals["self"], plot=None)

    # get the monitor's data
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if callback_vars["plot"] is None:  # make the plot
        plt.ion()
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        line, = ax.plot(x, y)
        callback_vars["plot"] = (line, ax, fig)
        plt.show()
    else:  # update and rescale the plot
        callback_vars["plot"][0].set_data(x, y)
        callback_vars["plot"][-2].relim()
        callback_vars["plot"][-2].set_xlim([_locals["total_timesteps"] * -0.02,
                                            _locals["total_timesteps"] * 1.02])
        callback_vars["plot"][-2].autoscale_view(True, True, True)
        callback_vars["plot"][-1].canvas.draw()


def compose_callbaclnk(*callback_funcs): # takes a list of functions, and returns the composed function.
    def _callback(_locals, _globals):
        continue_training = True
        for cb_func in callback_funcs:
            if cb_func(_locals, _globals) is False: # as a callback can return None for legacy reasons.
                continue_training = False
        return continue_training
    return _callback


################ TRAINING

model.learn(total_timesteps=time_steps, callback=auto_save_callback, seed=args.random_seed)

print('total time', time.time()-start)