# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

	def __init__(self, batch_size, seq_length, vocabulary_size,
				 lstm_num_hidden=256, lstm_num_layers=2, device="cuda:0"):

		super(TextGenerationModel, self).__init__()

		self.seq_length = seq_length
		self.batch_size = batch_size
		self.vocab_size = vocabulary_size

		self.lstm = nn.LSTM(input_size=vocabulary_size, 
							hidden_size=lstm_num_hidden, 
							num_layers=lstm_num_layers, 
							batch_first=True)

		self.fc = nn.Linear(in_features = lstm_num_hidden, 
							out_features =vocabulary_size,
							bias=True)
		self.to(device)
		
	def forward(self, x):
		# Output sequence
		# output_seq = torch.empty((self.seq_length, self.batch_size, self.vocab_size))
		# Pass in the input batch: B x S x D
		lstm_out, _ = self.lstm(x)
		# print("lstm out:",lstm_out.size())

		fc_out= self.fc(lstm_out)
		# print("fc out:", fc_out.size())

		# return output_seq.view((self.seq_length * self.batch_size, -1))
		return fc_out
