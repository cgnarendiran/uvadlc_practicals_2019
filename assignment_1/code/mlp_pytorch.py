"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch




class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layer and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layer, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(MLP, self).__init__()

    # [3072, 200, 200, 10]
    neuron_list = [n_inputs]+n_hidden+[n_classes]
    # [(3072,200), (200,200), (200,10)]
    neuron_pairs = [[neuron_list[i],neuron_list[i+1]] for i in range(len(neuron_list)-1)]

    # to store the modules
    layer_list = []


    for n,nps in enumerate(neuron_pairs):
      # last layer check
      if n+1 == len(neuron_pairs):
        layer_list.append(torch.nn.Linear(*nps))
        # layer_list.append(torch.nn.Softmax())
      # normally linear then relu
      else:
        layer_list.append(torch.nn.Linear(*nps))
        layer_list.append(torch.nn.ReLU())

    self.layers = torch.nn.Sequential(*layer_list)

    return
    
    
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.layers(x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
