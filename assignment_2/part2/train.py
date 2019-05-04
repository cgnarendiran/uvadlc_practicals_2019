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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def generate_sentence(model, dataset, device):
    start_char = np.random.randint(dataset.vocab_size)
    sentence = dataset._ix_to_char[start_char]
    one_hot = torch.rand(1, 1, dataset.vocab_size).to(device)
    one_hot[:, :, start_char] = 1   

    for i in range(config.seq_length - 1):
        out = model.forward(one_hot)
        test = out.squeeze()
        # print("test:", test.size())
        test3 = torch.softmax(test, dim=0)
        values, indices = torch.max(test3, 0)
        sentence += dataset._ix_to_char[indices.item()]
        one_hot = test3.view(1,1,dataset.vocab_size)

    return sentence

def index_to_onehot(batch_inputs, vocab_size):
    # batch_inputs = batch_inputs.type(torch.LongTensor).view(config.batch_size, config.seq_length, 1)
    # x_onehot = torch.FloatTensor(config.batch_size, config.seq_length, dataset._vocab_size) 
    # x_onehot.zero_()
    # x_onehot.scatter_(2, batch_inputs, 1)

    encodded_size = list(batch_inputs.shape)
    encodded_size.append(vocab_size)
    x_onehot = torch.zeros(encodded_size,
                          device=batch_inputs.device)
    x_onehot.scatter_(2, batch_inputs.unsqueeze(-1), 1)
    
    return x_onehot


def train(config):

    # Initialize the device which to run the model on
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    #path to save the model
    path = "results/"
    
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    # print("Data file:", dataset._data[0:5])
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset, config.lstm_num_hidden, config.lstm_num_layers, device)
    # model = torch.load("results/book_EN_grimms_fairy_tails_final_model.pt")

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr = config.learning_rate)

    # Store Accuracy and losses:
    results = {'accuracy': [], 'loss': []}

    # Training:
    total_steps = 0
    while total_steps <= config.train_steps:

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()
            optimizer.zero_grad()

            # Stacking and One-hot encoding:
            batch_inputs = torch.stack(batch_inputs,dim=1).to(device)
            batch_targets = torch.stack(batch_targets,dim=1).to(device)
            # print("Inputs and targets:", x_onehot.size(), batch_targets.size())

            # forward inputs to the model:
            pred_targets, _ = model.forward(index_to_onehot(batch_inputs, dataset.vocab_size))
            # print("pred_targets trans shape:", pred_targets.transpose(2,1).size())
            loss = criterion(pred_targets.transpose(2,1), batch_targets)

            #Backward pass
            loss.backward(retain_graph =True)
            optimizer.step()

            #Accuracy
            # argmax along the vocab dimension
            accuracy = (pred_targets.argmax(dim=2) == batch_targets).float().mean().item() 

            #Update the accuracy and losses for visualization:
            results['accuracy'].append(accuracy)
            results['loss'].append(loss.item())

            # Just for time measurement
            t2 = time.time()
            # examples_per_second = config.batch_size/float(t2-t1)
            total_steps += 1

            if step % config.print_every == 0:

                # print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                #       "Accuracy = {:.2f}, Loss = {:.3f}".format(
                #         datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                #         config.train_steps, config.batch_size, examples_per_second,
                #         accuracy, loss
                # ))
                print("[{}] Train Step {:07d}/{:07d}, Batch Size = {}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        total_steps, config.batch_size, results['accuracy'][-1], results['loss'][-1]
                ))

            if step%config.sample_every ==0:
                # Generate some sentences by sampling from the model
                print('GENERATED NO TEMP:')
                print(model.generate_sentence(100))
                print('__________________')
                print('GENERATED 0.5 TEMP:')
                print(model.generate_sentence(100, 0.5))
                print('__________________')
                print('GENERATED 1 TEMP:')
                print(model.generate_sentence(100, 1))
                print('__________________')
                print('GENERATED 2 TEMP:')
                print(model.generate_sentence(100, 2))
                # save model for individual timesteps
                torch.save(model, path + config.txt_file.split('/')[1].split('.')[0] + str(step) + "_model.pt")

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

        print('Done training.')
        #Save the final model
        
        torch.save(model, path + config.txt_file.split('/')[1].split('.')[0] + "_final_model.pt")
        print("saving results in folder...")
        np.save(path + "loss_train", results['loss'])
        np.save(path + "accuracy_train", results['accuracy'])



 ################################################################################
 ################################################################################
def read_data():
    dataset = TextDataset(config.txt_file, config.seq_length)
    print("Data file:", dataset._chars)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        print('batch_inputs:',batch_inputs)
        print('batch_targets:', batch_targets)
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=50, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
    # read_data()
