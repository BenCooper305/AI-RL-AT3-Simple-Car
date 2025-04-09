import gym
import simple_driving
#import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random
import matplotlib.pyplot as plt

alpha = 0.3
gamma = 0.95
num_episodes = 2000
epsilon = 0.2

Dubug = False

START_EPS = 0.9
MIN_EPS = 0.05
EPS_DECAY = 0.995

x_bins = np.linspace(-20, 20, 20)  # 5 bins for x-axis
y_bins = np.linspace(-20, 20, 20)  # 5 bins for y-axis
bins = [x_bins, y_bins]  # List of bins for each dimension

max_steps = 60

rewards_Over_timer = [] 

def plotRewardOverTime():
    y = rewards_Over_timer.copy()
    x_vals = list(range(len(y)))
    plt.plot(x_vals, y, marker='o')
    plt.xlabel("Index (x)")
    plt.ylabel("Value (y)")
    plt.title("Plot of y-values")
    plt.grid(True)
    plt.show()


def discretize_state(state):
    return tuple(np.digitize(state[i], bins[i]) - 1 for i in range(len(state)))   

def select_action(env, state, Q, episode):
    sample = np.random.uniform(0, 1)
    eps_threshold = MIN_EPS + (START_EPS * (EPS_DECAY * episode))
    if sample > eps_threshold:
        return np.argmax(Q[discretize_state(state)][:])
    else:
        return env.action_space.sample()

def train():
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    for episode in range(num_episodes):
        state, _ = env.reset()

        done = False

        for step in range(max_steps):
            action = select_action(env,state,q_table,episode)
            
            next_state, reward,done, truncated, info = env.step(action)


            old_value = q_table[discretize_state(state)][action]
            
            next_max = np.max(q_table[discretize_state(next_state)][:])
            
            q_table[discretize_state(state)][action] = (1- alpha) * old_value + alpha * (reward + gamma * next_max)
            

            if Dubug:
                print('ACTION ', action)
                print('raw  next state ', next_state)
                print('proc next state ', discretize_state(next_state))
                print('old value at state', state, 'with action', action,' ', old_value)
                print('max value for next state ', next_max)
                print('now the value for state ', state, ' with action ' , action, ' is ',  q_table[discretize_state(state)][action])


            state = next_state
            if done or truncated:
                print('episode: ', episode, ' with redward: ', reward)
                rewards_Over_timer.append(reward)
                break
    return q_table

def saveModel(model):
    torch.save(model,"q_values.pth")

def load():
    model = torch.load("q_values.pth")
                   
def runModel(q_table):
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
    env = env.unwrapped

    for episode in range(1):
        state, _ = env.reset()
        done = False

        print("Episode: ")
        print(episode)

        for step in range(max_steps):
            action = np.argmax(q_table[discretize_state(state)][:])
            print(action)
            next_state, reward,done, truncated, info = env.step(action)
            state = next_state

            if done or truncated:
                print('episode finished with reward: ', reward)
                break
    env.close()
            
def main():  
    q = train()
    plotRewardOverTime()
    runModel(q)

def simpleRun():
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
    env = env.unwrapped

    state, info = env.reset()

    for i in range(200):
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
        if done:
            break


#simpleRun()
main()
