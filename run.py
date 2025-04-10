import gym
import simple_driving
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random


# #create enviroment
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
# env = env.unwrapped

# #select run type
# simpleRun = False

# #set values for q-leanring
# gamma = 0.95                # discount factor - determines how much to value future actions
# episodes = 10000            # number of episodes to play out
# max_episode_length = 200    # maximum number of steps for episode roll out
# epsilon = 0.2               # control how often you explore random actions versus focusing on high value state and actions
# step_size = 0.1             # learning rate - controls how fast or slow to update Q-values for each iteration.

# #Modify the reward function (10%)
# def simulate(env, Q, max_episode_length, epsilon, episodes, episode):
#     D = []
#     state, _ = env.reset()                                                  
#     done = False
#     for step in range(max_episode_length):                                 
#         action = epsilon_greedy(env, state, Q, epsilon, episodes, episode)
#         next_state, reward, terminated, truncated, info = env.step(action)   
#         done = terminated or truncated 
#         #### change reward so that a negative reward is given if the agent falls down a hole #######
#         # if env.desc[int(next_state/env.ncol), int(next_state % env.ncol)] == b'H':  # fell in hole
#         #     reward = -1.0
#         ############################################################################################
#         D.append([state, action, reward, next_state])                       
#         state = next_state                                                  
#         if done:                                                            # if we fall into a hole or reach treasure then end episode
#             break
#     return D   

# def q_learning(env, gamma, episodes, max_episode_length, epsilon, step_size):
#     Q = defaultdict(lambda: np.zeros(env.action_space.n))                       
#     total_reward = 0
#     for episode in range(episodes):                                             
#         D = simulate(env, Q, max_episode_length, epsilon, episodes, episode)    
#         for data in D:                                                          
#             state = data[0]
#             action = data[1]
#             reward = data[2]
#             next_state = data[3]
#             Q[state][action] = (1 - step_size) * Q[state][action] + step_size * (reward +  gamma * np.max(Q[next_state])) 
#             total_reward += reward
#         if episode % 100 == 0:
#             print("average total reward per episode batch since episode ", episode, ": ", total_reward/ float(100))
#             total_reward = 0
#     return Q 

# #Modify the epsilon-greedy function to incorporate prior knowledge (20%)
# def epsilon_greedy(env, state, Q, epsilon, episodes, episode):
#     EPS_START = 0.9
#     EPS_END = 0.05
#     EPS_DECAY = episodes
#     sample = np.random.uniform(0, 1)
#     eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#         math.exp(-1. * episode / EPS_DECAY)
#     if sample > eps_threshold:
#         return np.argmax(Q[state])
#     else:
#         return env.action_space.sample()

# if simpleRun == True:
#     print("Running Simple")
#     state, info = env.reset()
#     for i in range(200):
#         action = env.action_space.sample()
#         state, reward, done, _, info = env.step(action)
#         if done:
#             break
# else:
#     print("Running Q-Leanring")
#     steps = 0
#     Q = q_learning(env, gamma, episodes, max_episode_length, epsilon, step_size)
#     while True and steps < 1000:
#         action = np.argmax(Q[state]) 
#         state, reward, terminated, truncated, info = env.step(action)  
#         done = terminated or truncated 
#         if done:
#             break

# env.close()


import numpy as np

print(np.__version__)