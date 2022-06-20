import numpy as np
from stable_baselines.results_plotter import ts2xy, load_results
import matplotlib.pyplot as plt
from stable_baselines import results_plotter

############ load data
possensor = ''
lr005=''

def moving_average(values,window):
    weights = np.repeat(1.0,window)/window
    return np.convolve(values,weights,'valid')

def plot_results(log_folder,title='learning curve'):
    x,y = ts2xy(load_results(log_folder),'timesteps')
    y = moving_average(y,window=50)

    x = x[len(x)-len(y)]

    fig = plt.figure(title)
    plt.plot(x,y)
    plt.xlabel('number of timesteps')
    plt.ylabel('rewards')
    plt.title(title+'smoothed')
    # plt.show()

def plot_successrate(log_folder,model,title="success rate"):
    x,y=ts2xy(load_results(log_folder),'timesteps')
    n=0
    episodes=[]
    for i in range(len(x)):
        episodes.append(n)
        n+=1
    successrate=[]
    nrofgrasps =0 
    j=0
    for i in y:
        if i > 900: # why 900
            nrofgrasps +=1
        j+=1
        successrate.append(nrofgrasps/j)

    fig = plt.figure(title)
    plt.plot(episodes,successrate,lable=model)
    plt.xlable('episodes')
    plt.ylable('success rate')
    plt.title(title)

plot_successrate(possensor,'Possensor')

plt.legend()
plt.show()

plot_results(possensor)
plt.legend()
plt.show()
