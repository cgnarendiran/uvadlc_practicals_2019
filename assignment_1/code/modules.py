"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    mu = 0
    sigma = 0.0001
    self.params['weight'] = np.random.normal(mu, sigma, (out_features, in_features))
    self.params['bias'] = np.zeros((out_features,1))
    self.grads['weight'] = np.zeros((out_features, in_features))
    self.grads['bias'] = np.zeros((out_features,1))
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = np.asarray(x)
    ### sgd input x: 1 x dl-1
    # out = self.params['weight'] @ self.x  + self.params['bias']

    ### batch input x: B x dl-1
    out = self.params['weight'] @ self.x.T  + self.params['bias']
    out = out.T
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # dx = dout @ self.params['weight'] 
    # self.grads['weight'] = dout.T @ self.x
    # self.gards['bias'] = dout @ np.eye(dout.shape[0])

    ### batch dout: B x dl
    dx = dout @ self.params['weight'] 
    self.grads['weight'] = dout.T @ self.x
    self.grads['bias'] = np.sum(dout, axis=0).reshape(self.grads['bias'].shape)
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = np.asarray(x)
    out = x*(x>0)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # dx = dout @ np.diag(self.x>0).astype(float)

    dx = dout * (self.x>0).astype(float)
    ########################
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # b = np.max(x)
    # xr = np.exp(x - b)
    # xr_max = np.sum(xr)
    # out = xr/xr_max
    # self.out = out

    ## shape of out is B x 10
    b = np.max(x,axis=1).reshape(x.shape[0],-1)
    xr = np.exp(x - b)
    xr_max = np.sum(xr,axis=1).reshape(xr.shape[0],-1)
    out = xr/xr_max
    self.out = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # s = self.out.reshape((-1,1))
    # dx_dx_tilde = np.diagflat(self.out) - np.dot(s, s.T)
    # dx = dout @ dx_dx_tilde

    ## sm_II:
    diag = np.zeros((self.out.shape[0],self.out.shape[1],self.out.shape[1]))
    ii = np.arange(self.out.shape[1])
    diag[:,ii,ii] = self.out

    outer = np.einsum('ij, ik -> ijk', self.out, self.out)
    dx_dx_tilde = diag - outer

    dx =  np.einsum('ij, ijk -> ik', dout, dx_dx_tilde)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    epsilon = 1e-10
    out = -(1/x.shape[0]) *np.sum(y*np.log(x+epsilon))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    epsilon = 1e-10
    dx = -(1/x.shape[0]) *y*(1/(x+epsilon))

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
