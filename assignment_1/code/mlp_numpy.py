"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
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

    # [3072, 200, 200, 10]
    neuron_list = [n_inputs]+n_hidden+[n_classes]
    # [(3072,200), (200,200), (200,10)]
    neuron_pairs = [[neuron_list[i],neuron_list[i+1]] for i in range(len(neuron_list)-1)]

    # to store the modules
    self.layers = []
    self.activations = []


    for n,nps in enumerate(neuron_pairs):

      # last layer check
      if n+1 == len(neuron_pairs):
        self.layers.append(LinearModule(*nps))
        self.activations.append(SoftMaxModule())
      # normally linear then relu
      else:
        self.layers.append(LinearModule(*nps))
        self.activations.append(ReLUModule())
    
        

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
    for l in range(len(self.layers)):
      if l == 0:
        z = self.layers[l].forward(x)
      else:
        z = self.layers[l].forward(a)
      a = self.activations[l].forward(z)

    # output from softmax layer
    out = a

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    for l in range(len(self.layers)-1,-1,-1):
      act_dout = self.activations[l].backward(dout)
      dout = self.layers[l].backward(act_dout)
    ########################
    # END OF YOUR CODE    #
    #######################

    return
