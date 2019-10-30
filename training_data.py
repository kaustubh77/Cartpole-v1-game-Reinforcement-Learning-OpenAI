'''
  training_data.py
  This module obtains all necessary training data.
  This includes the initial training data and any further data.
'''
import gym
import random
import numpy as np
from statistics import mean, median
from collections import Counter
from random import shuffle

env = gym.make('CartPole-v1')
env.reset()
goal_steps = 500
# The minimum score which the random games played should achieve in order to be recorded
score_requirement = 50
# The number of initial games to play to obtain a training sample can be tweaked
# Set to 15000 to try and get an accurate sample without spending too much time
initial_games = 15000

# Obtain an initial sample of training data by play x number of games with random values
# As the input
def initial_training_data():
  training_data = []
  scores = []
  accepted_scores = []
  for _ in range(initial_games):
    score = 0
    game_memory = []
    prev_observation = []
    for _ in range(goal_steps):
      action = random.randrange(0, 2)
      observation, reward, done, info = env.step(action)
    
      if len(prev_observation) > 0:
        game_memory.append([prev_observation, action])

      prev_observation = observation
      score += reward
      if done: 
        env.reset()
        break

    if score >= score_requirement:
      accepted_scores.append(score)
      for data in game_memory:
        if data[1] == 1:
          output = [0,1]
        elif data[1] == 0:
          output = [1, 0]
        training_data.append([data[0], output])

    env.reset()
    scores.append(score)
  
  print('Average accepted score: ', mean(accepted_scores))
  print('Median accepted score: ', median(accepted_scores))
  print(Counter(accepted_scores))

  return training_data

# Obtain any further training samples, where the model will play some of the games
# And there will be a bunch of random games as well
# The numbers can be changed. However, when playing the games by the model,
# It will take some time because the model is actually playing the game, it's not just random values
def further_training_data(model):
  training_data = []
  scores = []
  accepted_scores = []

  # Games played by the model
  for _ in range(500):
    score = 0
    game_memory = []
    prev_observation = []
    for _ in range(goal_steps):
      if len(prev_observation) > 0:
        action = np.argmax(model.predict(prev_observation.reshape(-1,len(prev_observation)))[0])
      else:
        action = random.randrange(0, 2)
      observation, reward, done, info = env.step(action)
    
      if len(prev_observation) > 0:
        game_memory.append([prev_observation, action])

      prev_observation = observation
      score += reward
      if done:
        env.reset()
        break

    if score >= score_requirement:
      accepted_scores.append(score)
      for data in game_memory:
        if data[1] == 1:
          output = [0,1]
        elif data[1] == 0:
          output = [1, 0]

      training_data.append([data[0], output])

    env.reset()
    scores.append(score)
  
  # Games played with random values
  for _ in range(4500):
    score = 0
    game_memory = []
    prev_observation = []
    for _ in range(goal_steps):
      action = random.randrange(0, 2)
      observation, reward, done, info = env.step(action)
    
      if len(prev_observation) > 0:
        game_memory.append([prev_observation, action])

      prev_observation = observation
      score += reward
      if done: 
        env.reset()
        break

    if score >= score_requirement:
      accepted_scores.append(score)
      for data in game_memory:
        if data[1] == 1:
          output = [0,1]
        elif data[1] == 0:
          output = [1, 0]

        training_data.append([data[0], output])

    env.reset()
    scores.append(score)

  print('Average accepted score: ', mean(accepted_scores))
  print('Median accepted score: ', median(accepted_scores))
  print(Counter(accepted_scores))

  # Shuffle the data just to prevent any overfitting
  shuffle(training_data)
  return training_data
