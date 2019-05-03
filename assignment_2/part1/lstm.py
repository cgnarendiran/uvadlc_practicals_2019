################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.device = device

        dist = torch.distributions.normal.Normal(0, 0.001)

        # Parameters for g
        self.W_gx = nn.Parameter(dist.sample(sample_shape=(num_hidden, input_dim)))
        self.W_gh = nn.Parameter(dist.sample(sample_shape=(num_hidden, num_hidden)))
        self.b_g = nn.Parameter(dist.sample(sample_shape=(num_hidden,)))

        # Parameters for i
        self.W_ix = nn.Parameter(dist.sample(sample_shape=(num_hidden, input_dim)))
        self.W_ih = nn.Parameter(dist.sample(sample_shape=(num_hidden, num_hidden)))
        self.b_i = nn.Parameter(dist.sample(sample_shape=(num_hidden,)))

        # Parameters for f
        self.W_fx = nn.Parameter(dist.sample(sample_shape=(num_hidden, input_dim)))
        self.W_fh = nn.Parameter(dist.sample(sample_shape=(num_hidden, num_hidden)))
        self.b_f = nn.Parameter(dist.sample(sample_shape=(num_hidden,)))

        # Parameters for o
        self.W_ox = nn.Parameter(dist.sample(sample_shape=(num_hidden, input_dim)))
        self.W_oh = nn.Parameter(dist.sample(sample_shape=(num_hidden, num_hidden)))
        self.b_o = nn.Parameter(dist.sample(sample_shape=(num_hidden,)))

        # Parameters for the linear layer
        self.W_ph = nn.Parameter(dist.sample(sample_shape=(num_classes, num_hidden)))
        self.b_p = nn.Parameter(dist.sample(sample_shape=(num_classes,)))

        self.to(device)

        

    def forward(self, x):
        # Implementation here ...
        # Assuming x is (B x T x input_dim)
        # h: (B x num_hidden)
        # p: (B x num_classes)

        c = torch.zeros(self.batch_size, self.num_hidden, device = self.device)
        h = torch.zeros(self.batch_size, self.num_hidden, device = self.device)
        

        for t in range(self.seq_length):	
        	x_t = x[:,t].view(-1,self.input_dim)
        	g = (x_t @ self.W_gx.t() +  h @ self.W_gh.t() + self.b_g).tanh()
        	i = (x_t @ self.W_ix.t() +  h @ self.W_ih.t() + self.b_i).sigmoid()
        	f = (x_t @ self.W_fx.t() +  h @ self.W_fh.t() + self.b_f).sigmoid()
        	o = (x_t @ self.W_ox.t() +  h @ self.W_oh.t() + self.b_o).sigmoid()
        	
        	c = g * i + c * f
        	h = c.tanh() * o

        p = h @ self.W_ph.t() + self.b_p
        return p
        