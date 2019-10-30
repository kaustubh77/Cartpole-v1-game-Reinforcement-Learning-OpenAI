'''
  model.py
  Define the neural network using Keras
  Train the model by obtaining data from training_data
  Test the model by playing 100 games and outputting the average score
'''
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Input

from training_data import env
from training_data import goal_steps

def neural_network_model(input_size):
  # Creates a model with 11 layers in total
  # Dropout layers are added to prevent overfitting
  model = Sequential()
  
  model.add(Dense(128, input_shape=(input_size,)))
  model.add(Dropout(0.2))

  model.add(Dense(256, activation=tf.nn.relu))
  model.add(Dropout(0.2))

  model.add(Dense(512, activation=tf.nn.relu))
  model.add(Dropout(0.2))

  model.add(Dense(256, activation=tf.nn.relu))
  model.add(Dropout(0.2))

  model.add(Dense(128, activation=tf.nn.relu))
  model.add(Dropout(0.2))

  model.add(Dense(2, activation=tf.nn.softmax))

  model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])
  return model

def train_model(training_data, model=False):
  # Convert the input values into Numpy arrays for training the model
  X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
  y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))

  if not model:
    model = neural_network_model(input_size = len(X[0]))
  
  # Number of epochs and validation split can be changed. However,
  # Increasing the epochs did not really do much to the loss and accuracy, so 3 is a good spot
  model.fit(X, y, epochs=5, validation_split=0.3)
  
  return model

def test_model(model):
  # Play the game 100 times and keep track of the score attained
  # The model has to achieve an average score over 195 for 100 trials to be considered successful

  scores = []
  choices = []

  for each_game in range(100):
      # env.render()
      score = 0
      game_memory = []
      previous_observation = []
      env.reset()
      for _ in range(goal_steps):
  
          # If it is the first move of the trial, make a random move
          if len(previous_observation)==0:
              action = random.randrange(0,2)
          else:
              action = np.argmax(model.predict(previous_observation.reshape(-1,len(previous_observation)))[0])
  
          choices.append(action)
                  
          new_observation, reward, done, info = env.step(action)
          previous_observation = new_observation
          game_memory.append([new_observation, action])
          score+=reward

          if done:
            env.reset()
            break
  
      scores.append(score)
  
  # Print out for user feedback
  print('Average Score:',sum(scores)/len(scores))
  model_name = 'ACC'+str(sum(scores)/len(scores))+'.model'
  model.save(model_name)
  print('Choice 0:{0} | Choice 1:{1}'.format(choices.count(0)/len(choices),choices.count(1)/len(choices)))
