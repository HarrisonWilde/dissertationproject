import argparse
import math
import os
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from datetime import datetime
from tqdm import tqdm

try:
    import matplotlib
    if os.environ.get('DISPLAY','') == '':
        print('no display found. Using non-interactive Agg backend')
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except:
    print("matplotlib doesn't work on Mac OS")

import config


"""
Trains a model on multiple seq batches by iterating through a generator.
"""
def train(model, training_batch_generator, training_steps, validation_batch_generator, validation_steps, optimiser, criterion):

    epoch = 1

    try:
        training_losses = np.load('training_losses.npy').tolist()
        validation_losses = np.load('validation_losses.npy').tolist()
    except:
        print("Starting with fresh train and validation loss lists")
        training_losses = []
        validation_losses = []

    while True:
        
        """
        Training epoch
        """
        step = 1
        total_loss = 0

        with tqdm(range(training_steps)) as t:
            t.set_description('Epoch {}'.format(epoch))
            
            for _ in t:
                data = training_batch_generator()
                loss = train_step(model, data, optimiser, criterion)

                total_loss += loss
                avg_loss = total_loss / step
                t.set_postfix(loss=avg_loss)

                step += 1

        training_losses.append(avg_loss)

        """
        Validation epoch
        """
        step = 1
        total_loss = 0

        with tqdm(range(validation_steps)) as t:
            t.set_description('Validation {}'.format(epoch))

            for _ in t:
                data = validation_batch_generator()
                model.eval()
                loss = compute_loss(model, data, volatile=True)[1]
                total_loss += loss
                avg_loss = total_loss / step
                t.set_postfix(loss=avg_loss)
                step += 1
            
        validation_losses.append(avg_loss)


        """
        Save model params and results from the overall epoch
        """
        save_loss(training_losses, validation_losses, 'loss_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.png')
        torch.save(model.state_dict(), config.MODELS_DIR + '/model_' + str(epoch) + '.pt')
        epoch += 1


"""
Trains the model on a single batch of sequence.
"""
def train_step(model, data, optimiser, criterion):
    
    model.train()
    loss, avg_loss = compute_loss(model, data, criterion)
    loss = loss * config.SCALE_FACTOR

    # Zero the model's gradients
    model.zero_grad()
    loss.backward()

    # Copy the gradients calculated for the model into param_copy
    param_copy = model.param_copy
    for parameter, new_parameter in zip(param_copy, list(model.parameters())):
        if parameter.grad is None:
            parameter.grad = torch.nn.Parameter(parameter.data.new().resize_(*parameter.data.size()))
        parameter.grad.data.copy_(new_parameter.grad.data)

    # Unscale the gradient
    if config.SCALE_FACTOR != 1:
        for parameter in param_copy:
            parameter.grad.data /= config.SCALE_FACTOR

    # clip_grad_norm_ helps prevent the exploding gradient problem in RNNs / LSTMs.
    # Reference: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    clip_grad_norm_(model.parameters(), config.CLIP_FACTOR)
    optimiser.step()

    # Copy the parameters back into the model
    model_params = list(model.parameters())
    for i in range(len(param_copy)):
        model_params[i].data.copy_(param_copy[i].data)

    return avg_loss


"""
Trains the model on a single batch of sequence.
"""
def compute_loss(model, data, criterion, volatile=False):

    # Declare all variables by converting from the passed tensors
    note_seq, moods = data
    moods = Variable(moods)
    # One hot the inputs
    inputs = Variable(torch
        .FloatTensor(note_seq[:, :-1].size(0), note_seq[:, :-1].size(1), config.FULL_RANGE)
        .zero_()
        .scatter_(2, index_batch.unsqueeze(2), 1.0))
    targets = Variable(note_seq[:, 1:])
    if torch.cuda.is_available():
        moods = moods.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()

    # Input data to the model and then calculate the corresponding loss values using CrossEntropyLoss.
    if volatile:
        with torch.no_grad():
            outputs, _ = model(inputs, moods, None)
            loss = criterion(outputs.view(-1, config.FULL_RANGE).float(), targets.contiguous().view(-1))
    else:
        outputs, _ = model(inputs, moods, None)
        # print(outputs.view(-1, config.FULL_RANGE).float()[0][targets.contiguous().view(-1)[0]])
        # print(outputs.view(-1, config.FULL_RANGE).float())
        # print(targets.contiguous().view(-1))
        loss = criterion(outputs.view(-1, config.FULL_RANGE).float(), targets.contiguous().view(-1))

    return loss, loss.data.item()


"""
Saves arrays of loss values and plots a graph each epoch of these values
"""
def save_loss(training_loss, validation_loss, name):

    np.save('training_losses.npy', np.array(training_losses))
    np.save('validation_losses.npy', np.array(validation_losses))
    plt.clf()
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.savefig(config.GRAPHS_DIR + '/' + name)
