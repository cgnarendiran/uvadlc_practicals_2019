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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

import matplotlib.pyplot as plt

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################
def visualize(x,  results_rnn, results_lstm):
    plt.style.use('seaborn-darkgrid')
    plt.figure(0)
    plt.plot(np.int_(x), results_rnn,  marker='o', linewidth=3, markersize=8, label='RNN')
    plt.plot(np.int_(x), results_lstm,  marker='s',  linewidth=1, label='LSTM')
    plt.title('Accuracy vs Sequence length of the RNN vs LSTM models')

    plt.xlabel('Sequence length of the Palindrome')
    plt.ylabel('Max Accuracy')
    plt.savefig('rnn_lstm.png')
    plt.legend()
    plt.show()


def train(config, pallindrome_length, m):

    
    config.input_length = pallindrome_length
    config.model_type = m
    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model that we are going to use
    hyper_params = [config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device]
    model = globals()['Vanilla'+config.model_type](*hyper_params) if config.model_type=='RNN' else globals()[config.model_type](*hyper_params)


    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)


    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr = config.learning_rate)

    accuracies = []
    losses = []
    avg_loss = 0



    
    ########## One hot encoding buffer that you create out of the loop and just keep reusing
    # if config.input_dim != 1:
    #     nb_digits = 10
    #     x_onehot = torch.FloatTensor(config.batch_size, config.input_length, nb_digits)    


    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        # Only for time measurement of step through network
        t1 = time.time()

        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        optimizer.zero_grad()

        # Forward pass:
        ########## Convert input to one-hot:
        # if config.input_dim != 1:
        #     batch_inputs = batch_inputs.type(torch.LongTensor).view(config.batch_size, config.input_length, 1)
        #     x_onehot.zero_()
        #     x_onehot.scatter_(2, batch_inputs, 1)
        #     y_pred = model.forward(x_onehot)
        # else:
        #     y_pred = model.forward(batch_inputs)

        y_pred = model.forward(batch_inputs)
        loss = criterion.forward(y_pred,batch_targets)

        #Backward pass
        
        loss.backward(retain_graph =True)

        optimizer.step()

        accuracy = (y_pred.argmax(dim=1) == batch_targets).float().mean().item()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)
        # accuracies.append(accuracy)
        losses.append(loss.item())

        if step % 500 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))
            accuracies.append(accuracy)
            if loss < 0.01 or accuracy == 1:
                break
            else:
                avg_loss = np.average(losses)
                losses = []

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    return max(accuracies)


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=5000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    # train(config,4)
    results_rnn = []
    results_lstm = []
    l = 4
    h = 30
    for i in range(l,h):
        print("Training RNN with pallindrome_length:",i)
        results_rnn.append(train(config, i, 'RNN'))
        print("Training LSTM with pallindrome_length:",i)
        results_lstm.append(train(config, i, 'LSTM'))

    visualize(np.arange(l,h), results_rnn, results_lstm)