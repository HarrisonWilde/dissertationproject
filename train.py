import os
import numpy as np
import torch
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
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

    try:

        training_losses = np.load('training_losses.npy').tolist()
        validation_losses = np.load('validation_losses.npy').tolist()
    
    except:

        print("Starting with fresh train and validation loss lists")
        training_losses = []
        validation_losses = []

    n_epochs = 1
    while True:
        
        """
        Epoch training
        """
        with tqdm(range(training_steps)) as p:

            p.set_description('Epoch {}'.format(n_epochs))
            loss_sum = 0
            for _ in p:

                data = training_batch_generator()
                loss = train_step(model, data, optimiser, criterion)
                p.set_postfix(loss=loss)
                loss_sum += loss

        training_losses.append(loss_sum / training_steps)

        """
        Epoch validation
        """
        with tqdm(range(validation_steps)) as p:

            p.set_description('Validation {}'.format(n_epochs))
            loss_sum = 0
            for _ in p:
            
                model.eval()
                data = validation_batch_generator()
                loss = input_batch(model, data, criterion, volatile=True)[1]
                p.set_postfix(loss=loss)
                loss_sum += loss
            
        validation_losses.append(loss_sum / validation_steps)

        """
        Save model params and results from the overall epoch
        """
        save_progress(training_losses, validation_losses, 'loss_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.png')
        torch.save(model.state_dict(), config.MODELS_DIR + '/model_' + str(n_epochs) + '.pt')
        n_epochs += 1


"""
Trains the model on a single batch of sequence.
"""
def train_step(model, data, optimiser, criterion):
    
    model.train()
    loss, avg_loss = input_batch(model, data, criterion)
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
    for parameter in param_copy:
        parameter.grad.data /= config.SCALE_FACTOR

    # clip_grad_norm_ mitigates exploding gradients in RNNs as discussed in the final report
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
def input_batch(model, data, criterion, volatile=False):

    # Declare all variables by converting from the passed tensors
    note_seq, moods = data
    moods = Variable(moods)
    # One hot the inputs
    inputs = Variable(torch
        .FloatTensor(note_seq[:, :-1].size(0), note_seq[:, :-1].size(1), config.FULL_RANGE)
        .zero_()
        .scatter_(2, note_seq[:, :-1].unsqueeze(2), 1.0))
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
Saves arrays of loss values and plots a graph each n_epochs of these values
"""
def save_progress(training_loss, validation_loss, name):

    np.save('training_losses.npy', np.array(training_losses))
    np.save('validation_losses.npy', np.array(validation_losses))
    plt.clf()
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.savefig(config.GRAPHS_DIR + '/' + name)
