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
import numpy as np


class TextGenerationModel(nn.Module):

	def __init__(self, batch_size, seq_length, dataset,
				 lstm_num_hidden=256, lstm_num_layers=2, device="cuda:0"):

		super(TextGenerationModel, self).__init__()

		self.seq_length = seq_length
		self.batch_size = batch_size
		self.dataset = dataset
		self.device = device

		self.lstm = nn.LSTM(input_size=self.dataset.vocab_size, 
							hidden_size=lstm_num_hidden, 
							num_layers=lstm_num_layers, 
							batch_first=True)

		self.fc = nn.Linear(in_features = lstm_num_hidden, 
							out_features =self.dataset.vocab_size,
							bias=True)
		self.to(device)
		
	def forward(self, x, h0=None, c0=None):
		# Output sequence
		# output_seq = torch.empty((self.seq_length, self.batch_size, self.vocab_size))
		# Pass in the input batch: B x S x D
		x = x.to(self.device)
		lstm_out, (h_n, c_n) = self.lstm(x)
		# print("lstm out:",lstm_out.size())

		fc_out= self.fc(lstm_out)
		# print("fc out:", fc_out.size())

		# return output_seq.view((self.seq_length * self.batch_size, -1))
		return fc_out, (h_n, c_n)

	def generate_sentence(self, seq_length, temperature=0.0):
		h_n,c_n = None,None
		last_char = np.random.randint(self.dataset.vocab_size)
		sentence = self.dataset._ix_to_char[last_char]

		for i in range(seq_length - 1):
			one_hot = torch.zeros(1, 1, self.dataset.vocab_size)
			one_hot[:, :, last_char] = 1
			out, (h_n, c_n) = self.forward(one_hot, h_n, c_n)
			out = out.squeeze()
			if temperature == 0:
				out = torch.softmax(out, dim=0)
				values, last_char = torch.max(out, 0)
			else:
				out = torch.softmax((1 / temperature) * out, dim=0)
				last_char = torch.multinomial(out, 1)
			sentence += self.dataset._ix_to_char[last_char.item()]

		return sentence
