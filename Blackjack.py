import pandas as pd
from collections import defaultdict
import gymnasium as gym
env = gym.make('Blackjack-v1',render_mode = 'human')

def policy(state):
    return 0 if state[0]>19 else 1

state = env.reset()
state = state[0]
print(state)

print(policy(state))
num_timesteps =100

def genetate_episode(policy):
    episode = []
    state = env.reset()
    state = state[0]
    for t in range (num_timesteps):
        action = policy(state)
        next_state, reward, done, info, trans_prob = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state= next_state
    return episode
print(genetate_episode(policy))

total_return = defaultdict(float)
N = defaultdict(int)
num_iterations = 50000
for i in range(num_iterations):
    episode = genetate_episode(policy)
    states, actions, rewards = zip(*episode)
    for t, state in enumerate(states):
        R = (sum(rewards[t:]))
        total_return[state] = total_return[state]+R
        N[state]= N[state]+1

print(total_return[state])
print(N[state])

total_return = pd.DataFrame(total_return.items(),columns=['state','total_return'])
N = pd.DataFrame(N.items(),columns=['state','N'])
df = pd.merge(total_return, N, on="state")
print(df.head(10))
df['value'] = df['total_return']/df['N']
print(df.head(10))
print(df.shape)










# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:55:27 2023

@author: susheel
"""

#Implementation of Every-visit Monte Carlo Prediction for Blackjack environment.

import pandas as pd
from collections import defaultdict
import gymnasium as gym
env=gym.make('Blackjack-v1',render_mode='human')

def policy(state):
    return 0 if state[0]>19 else 1

state = env.reset()
state=state[0]
print(state)

print(policy(state))

num_timesteps = 100

def generate_episode(policy):
    episode = []
    state = env.reset()
    state=state[0]
    for t in range(num_timesteps):
        action = policy(state)
        next_state, reward, done, info,trans_prob = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state=next_state
    return episode
print(generate_episode(policy))


total_return = defaultdict(float)
N = defaultdict(int)
num_iterations = 50000
for i in range(num_iterations):
    episode = generate_episode(policy)
    states, actions, rewards = zip(*episode)
    for t, state in enumerate(states):
        R = (sum(rewards[t:]))
        total_return[state] = total_return[state] + R
        N[state] = N[state] + 1
        
print(total_return[state])
print(N[state])
        
total_return = pd.DataFrame(total_return.items(),columns=['state', 'total_return'])
N = pd.DataFrame(N.items(),columns=['state', 'N'])
df = pd.merge(total_return, N, on="state")
print(df.head(10))
df['value'] = df['total_return']/df['N']
print(df.head(10))
print(df.shape)

df[df['state']==(21,9,False)]['value'].values
df[df['state']==(5,8,False)]['value'].values
        
