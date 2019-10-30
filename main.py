'''
  main.py
  Main file which will create the training data and train the model
'''
from training_data import initial_training_data
from training_data import further_training_data
from model import train_model
from model import test_model

def main():
  # Creates the initial training sample and trains the model
  # test_model will then make the model play games to determine how well it scores
  training_data = initial_training_data()
  model = train_model(training_data)
  test_model(model)

if __name__ == '__main__':
  main()
