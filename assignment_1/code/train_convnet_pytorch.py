"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import *
import cifar10_utils
import torchvision


# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None



#set datatype to torch tensor
dtype = torch.FloatTensor

#use GPUs if available
device =  torch.device('cuda') #  torch.device('cuda' if torch.cuda.is_available() else 'cpu') #


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
  accuracy = (predictions.argmax(dim=1) == targets).float().mean().item()
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)
  ####################################################

  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  #load test data
  x_test = cifar10["test"].images
  y_test = cifar10["test"].labels
  #reshape the test data so the MLP can read it.
  # x_test = x_test.reshape(x_test.shape[0],-1)
  #tranform np arrays into torch tensors
  x_test = torch.tensor(x_test, requires_grad=False).type(dtype).to(device)
  y_test = torch.tensor(y_test, requires_grad=False).type(dtype).to(device)
  
  #get the number of classes
  # n_classes = y_test.shape[1]
  
  #################################################
  # define the model
  convnet_model = ConvNet(x_test.shape[1], y_test.shape[1])
  loss_fn = nn.CrossEntropyLoss()
  optmizer = torch.optim.Adam(cnn_model.parameters(), lr=FLAGS.lr)

  cnn_model.to(device)
  
  #reshape y for torch suitability
  y_test = y_test.argmax(dim=1)
  #metrics to keep track during training.
  acc_train = []
  acc_test = []
  loss_train = []
  loss_test = []


  ########################################## train the data:
  for step in range(FLAGS.max_steps):
    # get the net batch
    # Mini-batch images and labels.

    # try:
    #   x, y = data_iter_train.next()
    #   x = x.reshape(x.shape[0],-1)

    # except StopIteration:
    #   data_iter_train = iter(train_loader)
    #   x, y = data_iter_train.next()
    #   x = x.reshape(x.shape[0],-1)

    ####################################
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    #reshape images into a vector
    # x = x.reshape(x.shape[0],-1)
    #use torch tensor + gpu 
    x = torch.from_numpy(x).type(dtype).to(device)
    y = torch.from_numpy(y).type(dtype).to(device)
    y = y.argmax(dim=1)
    ###################################

    # forward + backward + optimize
    y_pred = convnet_model.forward(x)
    # print(y_pred[:])
    # print(y[:])
    loss = loss_fn(y_pred, y)

    # zero the parameter gradients
    optimizer.zero_grad()


    loss.backward()
    optimizer.step()
    y_pred.detach()

    # Evaluate accuracies and losses:
    if (step%FLAGS.eval_freq==0)  or step==FLAGS.max_steps-1:
      print("step:",step)
      
      #keep metrics on training set:
      loss_train.append(loss.item())
      acc_train.append(accuracy(y_pred, y))
      print("train performance: acc = ", acc_train[-1], "loss = ", loss_train[-1])
      
      #keep metrics on the test set:
      with torch.no_grad():
        y_test_pred = convnet_model.forward(x_test)
        loss_test.append(loss_fn(y_test_pred, y_test).item())
        acc_test.append(accuracy(y_test_pred, y_test))
      print("test performance: acc = ", acc_test[-1], "loss = ",loss_test[-1])
      
      #
      if len(loss_train)> 10:
        if (np.mean(loss_train[-10:-5]) - np.mean(loss_train[-5:])) < 1e-7:
          print("Early Stopping")
          break  


  #store results after training:
  path = "../cnn_results/"
  print("saving results in folder...")
  np.save(path + "cnn_loss_train", loss_train)
  np.save(path + "cnn_accuracy_train", acc_train)
  np.save(path + "cnn_loss_test", loss_test)
  np.save(path + "cnn_accuracy_test", acc_test) 


  print("saving model in folder")
  torch.save(convnet_model.state_dict(), convnet_model.__class__.__name__ + ".pt")
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
