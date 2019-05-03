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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device):
        super(VanillaRNN, self).__init__()
        # Initialization here ...

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.device = device

        dist = torch.distributions.normal.Normal(0, 0.001)
        self.W_hx = nn.Parameter(dist.sample(sample_shape=(num_hidden, input_dim)))
        self.W_hh = nn.Parameter(dist.sample(sample_shape=(num_hidden, num_hidden)))
        self.W_ph = nn.Parameter(dist.sample(sample_shape=(num_classes, num_hidden)))
        self.b_h = nn.Parameter(dist.sample(sample_shape=(num_hidden,)))
        self.b_p = nn.Parameter(dist.sample(sample_shape=(num_classes,)))

        self.to(device)


    def forward(self, x):
        # Implementation here ...
        # Assuming x is (B x T x input_dim)
        # h: (B x num_hidden)
        # p: (B x num_classes)
        # torch.nn.init.xavier_uniform_(self.parameters())

        # init h1 to zero everytime for a new sequence
        h = torch.zeros(self.batch_size,self.num_hidden, device = self.device)
        
        for t in range(self.seq_length):
        	h = (x[:,t].view(-1,self.input_dim) @ self.W_hx.t() +  h @ self.W_hh.t() + self.b_h).tanh()
        # print(self.h.shape)

        p = h @ self.W_ph.t() + self.b_p
        # self.h.detach_()
        return p
        