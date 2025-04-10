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

alpha = 0.2
gamma = 0.95
num_episodes = 20000
epsilon = 0.2

Dubug = False

START_EPS = 0.9
MIN_EPS = 0.1
EPS_DECAY = 0.9998

NUM_BINS = 26
x_bins = np.linspace(-14, 14, NUM_BINS)  # 5 bins for x-axis
y_bins = np.linspace(-14, 14, NUM_BINS)  # 5 bins for y-axis
bins = [x_bins, y_bins]  # List of bins for each dimension

max_steps = 45

rewards_Over_time_train = [] 

def plotRewardOverTime(data):
    y = data.copy()
    x_vals = list(range(len(y)))
    plt.plot(x_vals, y, marker='o')
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("End Rewards vs Episode")
    plt.grid(True)
    plt.show()


def discretize_state(state):
    return tuple(np.digitize(state[i], bins[i]) - 1 for i in range(len(state)))   

def select_action(env, state, q_table, episode):
    eps_threshold = max(MIN_EPS, START_EPS * (EPS_DECAY ** episode))
    if np.random.uniform(0, 1) > eps_threshold:
        x = int(state[0])
        y = int(state[1])
        return np.argmax(q_table[x, y, :])
    else:
        return env.action_space.sample()

def train(q_table,env):
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = discretize_state(state)

        done = False

        for step in range(max_steps):
            action = select_action(env,state,q_table,episode)
            
            next_state, reward,done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state)

            old_value = q_table[state[0],state[1],action]
            
            next_max = np.max(q_table[next_state[0],next_state[1],:])

            q_table[state[0],state[1],action] = (1- alpha) * old_value + alpha * (reward + gamma * next_max)

            # next_max = np.max(q_table[discretize_state(next_state)][:])
            # old_value = q_table[discretize_state(state)][action]
            # q_table[discretize_state(state)][action] = (1- alpha) * old_value + alpha * (reward + gamma * next_max)
            
            if Dubug:
                print('ACTION ', action)
                print('raw  next state ', next_state)
                print('proc next state ', discretize_state(next_state))
                print('old value at state', state, 'with action', action,' ', old_value)
                print('max value for next state ', next_max)
                print('now the value for state ', state, ' with action ' , action, ' is ',  old_value)

            state = next_state
            if done or truncated:
                if Dubug:
                    print('episode: ', episode, ' with redward: ', reward)
                else:
                    if episode % 50 == 0:
                        print('episode: ', episode, ' out of ', num_episodes ,' episodes')
                rewards_Over_time_train.append(reward)
                break
    return q_table

def saveModel(model):
    print('file saved')
    torch.save(model,"q_values2.pth")

def loadQTable(env):
    try:
        q_table = torch.load("q_values2.pth")
        print("Model loaded successfully!")
        return q_table
    except Exception as e:
        print(f"Error loading model: {e}")
        q_table = np.zeros((NUM_BINS,NUM_BINS,env.action_space.n))
        return q_table  
                   
def runModel(q_table,iterations):
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
    env = env.unwrapped
    reachedgoal = 0
    rewards_test = []

    for episode in range(iterations):
        state, _ = env.reset()
        done = False

        for step in range(max_steps):
            x = int(state[0])
            y = int(state[1])
            action = np.argmax(q_table[x, y, :])
            next_state, reward,done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state)
            state = next_state

            if done or truncated:
                if Dubug:
                    print('episode: ', episode, ' with redward: ', reward)
                else:
                    if episode % 50 == 0:
                        print('episode: ', episode, ' out of ', iterations ,' episodes')
                rewards_test.append(reward)
                if reward > 40:
                    reachedgoal = reachedgoal + 1
                break
    plotRewardOverTime(rewards_test)
    print('the model reached goal ', reachedgoal, 'times ot of ', iterations)
    env.close()
            
def main():  
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
    q_table = loadQTable(env)
    q_table = train(q_table, env)
    plotRewardOverTime(rewards_Over_time_train)
    saveModel(q_table)

def simpleRun():
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
    env = env.unwrapped
    state, info = env.reset()

    for i in range(200):
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
        if done:
            break

def testElipson():
    for i in range(num_episodes):
        for j in range(max_steps):
            eps_threshold = max(MIN_EPS, START_EPS * (EPS_DECAY ** i))
            print(eps_threshold)
            
#simpleRun()
#main()
#testElipson()
q_table = torch.load("q_values2.pth")
#print(q_table)
runModel(q_table,2)