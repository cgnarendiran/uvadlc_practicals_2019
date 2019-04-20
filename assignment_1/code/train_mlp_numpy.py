"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  accuracy = np.mean(np.argmax(predictions,axis=1) == np.argmax(targets,axis=1))
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  # Load Cifar:
  cifar10 =  cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')

  #load test data
  x_test = cifar10["test"].images
  y_test = cifar10["test"].labels
  # reshape the test data
  x_test = x_test.reshape(x_test.shape[0],-1)

  # Reshape the train data input:
  # print(x.shape)
  # print(y.shape)
  # define the mlp model
  mlp_model = MLP(x_test.shape[1], dnn_hidden_units, y_test.shape[1])
  

  # finally loss
  loss_func = CrossEntropyModule()

  #metrics to keep track during training.
  acc_train = []
  acc_test = []
  loss_train = []
  loss_test = []


  # train the data:
  for step in range(FLAGS.max_steps):
    # get the net batch
    x , y = cifar10['train'].next_batch(FLAGS.batch_size)
    x = x.reshape(FLAGS.batch_size,-1)

    # Forward the data
    y_pred = mlp_model.forward(x)

    # Compute the loss:
    loss = loss_func.forward(y_pred,y)
    # Compute the loss gradient:
    loss_grad = loss_func.backward(y_pred,y)

    # Backward on the MLP module:
    mlp_model.backward(loss_grad)

    # Update the parameters:
    for layer in mlp_model.layers:
      layer.params["weight"] -= FLAGS.learning_rate*layer.grads["weight"]
      layer.params["bias"] -= FLAGS.learning_rate*layer.grads["bias"]

    



    # Evaluate accuracies and losses:
    if (step%FLAGS.eval_freq==0)  or step==FLAGS.max_steps-1:
      print("step:",step)
      
      #keep metrics on training set:
      loss_train.append(loss)
      acc_train.append(accuracy(y_pred, y))
      print("train performance: acc = ", acc_train[-1], "loss = ", loss_train[-1])
      
      #keep metrics on the test set:
      y_test_pred = mlp_model.forward(x_test)
      loss_test.append(loss_func.forward(y_test_pred, y_test))
      acc_test.append(accuracy(y_test_pred, y_test))
      print("test performance: acc = ", acc_test[-1], "loss = ",loss_test[-1])
      
      #
      if len(loss_train)> 10:
          if (np.mean(loss_train[-10:-5]) - np.mean(loss_train[-5:])) < 1e-6:
              print("Early Stopping")
              break  


  #store results after training:
  path = "../np_results/"
  print("saving results in folder...")
  np.save(path + "np_loss_train", loss_train)
  np.save(path + "np_accuracy_train", acc_train)
  np.save(path + "np_loss_test", loss_test)
  np.save(path + "np_accuracy_test", acc_test) 
  return

  ########################
  # END OF YOUR CODE    #
  #######################





def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()