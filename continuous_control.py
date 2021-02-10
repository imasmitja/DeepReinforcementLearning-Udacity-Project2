# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 06:19:54 2021

@author: Usuari
"""


"""
Continuous Control
You are welcome to use this coding environment to train your agent for the project. Follow the instructions below to get started!

1. Start the Environment
Run the next code cell to install a few packages. This line will take a few minutes to run!
"""
"""
The environments corresponding to both versions of the environment are already saved in the Workspace and can be accessed at the file paths provided below.

Please select one of the two options below for loading the environment.
"""

from unityagents import UnityEnvironment
import numpy as np
import gym
import time
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import progressbar as pb
import subprocess as sp
import os
from ddpg_agent import Agent
from collections import namedtuple, deque

def ddpg(env,agent,brain_name,n_episodes=300, max_t=2000, print_every=10):
    # widget bar to display progress
    widget = ['\rTraining loop: ', pb.Percentage(), ' ', 
              pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=n_episodes+1).start()
    
    scores_deque = deque(maxlen=100)
    scores_t = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment 
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        agent.reset()
        for t in range(max_t):
            actions = agent.act(states, add_noise=True)       # select an action (for each agent)
            env_info = env.step(actions)[brain_name]           # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            agent.step(states, actions, rewards, next_states, dones)
            scores += rewards                                  # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        scores_deque.append(np.mean(scores))
        scores_t.append(np.mean(scores))
        # print('\rEpisode {}\tAverage Score (averaged over agents): {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\r\nEpisode {}\tAverage Score (averaged over agents): {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if np.mean(scores_deque)>=100.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
        # update progress widget bar
        timer.update(i_episode+1)
    timer.finish()
    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
    return scores_t

#%%
# select this option to load version 1 (with a single agent) of the environment
# env = UnityEnvironment(file_name='1Agent/Reacher.exe')

# select this option to load version 2 (with 20 agents) of the environment
env = UnityEnvironment(file_name='20Agents/Reacher.exe')

"""
Environments contain brains which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.
"""
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

"""
2. Examine the State and Action Spaces
Run the code cell below to print some information about the environment.
"""
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

#%%
"""
3. Train the Agent with DDPG
"""
agent = Agent(state_size=state_size, action_size=action_size, random_seed=342)
scores = ddpg(env, agent,brain_name,n_episodes=300, max_t=2000, print_every=10)

#%%
"""
4. Plot and save the scores obtained
"""
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores))*10, scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

#%%
import pandas as pd
def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.figure()
    plt.plot(scores); plt.title("Reacher.exe");
    plt.xlabel('Episode #')
    plt.ylabel('Score')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);
    plt.show()
    return rolling_mean

#Visualitzation:
print("[TEST] Completed {} episodes with avg. score = {}".format(len(scores), np.mean(scores)))
rolling_mean = plot_scores(scores)

#Save scores
save_file = 'ddpg_t2'
np.savetxt(save_file+'_scores.txt', scores, delimiter = ',')

#%%
# #See a random agent
# env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
# states = env_info.vector_observations                  # get the current state (for each agent)
# scores = np.zeros(num_agents)                          # initialize the score (for each agent)
# while True:
#     actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
#     actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#     env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#     next_states = env_info.vector_observations         # get next state (for each agent)
#     rewards = env_info.rewards                         # get reward (for each agent)
#     dones = env_info.local_done                        # see if episode finished
#     scores += env_info.rewards                         # update the score (for each agent)
#     states = next_states                               # roll over states to next time step
#     if np.any(dones):                                  # exit loop if episode finished
#         break
# print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

#When finished, you can close the environment.
env.close()



























