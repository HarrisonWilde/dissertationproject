import os
import numpy as np
import torch
from datetime import datetime
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

try:
    import matplotlib
    if os.environ.get('DISPLAY','') == '':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except:
    print("matplotlib doesn't work on Mac OS")

import config


"""
Trains and validates a model, saving progress after each epoch
"""
def train(model, training_batch_generator, validation_batch_generator, optimiser, criterion, n_epochs):

    try:
        training_losses = np.load('training_losses.npy', allow_pickle=True).tolist()
        validation_losses = np.load('validation_losses.npy', allow_pickle=True).tolist()
    except:
        print("Starting with fresh train and validation loss lists")
        training_losses = []
        validation_losses = []

    session_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for epoch in range(n_epochs):
        
        loss = training_epoch(model, training_batch_generator, optimiser, criterion, epoch)
        training_losses.append(loss)

        loss = validation_epoch(model, validation_batch_generator, criterion, epoch)
        validation_losses.append(loss)

        save_progress(training_losses, validation_losses, '{}/loss_{}.png'.format(session_date, epoch))
        torch.save(model.state_dict(), config.MODELS_DIR + '/{}/model_{}.pt'.format(session_date, epoch))


"""
Set model to training mode, feed in training batches and re-calculate gradients using loss
"""
def training_epoch(model, training_batch_generator, optimiser, criterion, epoch):

    loss_sum = 0
    p = trange(config.TRAINING_STEPS, desc='T' + str(epoch))
    for _ in p:

        model.train()
        midi, moods = training_batch_generator()
        print(midi.size())

        loss = input_batch(model, midi, moods, criterion)
        loss_sum += loss.data.item()
        p.set_postfix(loss=loss.data.item())
        loss *= (midi.size(1) - 1)

        # Zero the model's gradients
        model.zero_grad()
        loss.backward()

        # Copy the gradients calculated for the model into previous_parameters
        previous_parameters = model.previous_parameters
        for previous_parameter, new_parameter in zip(previous_parameters, list(model.parameters())):
            if previous_parameter.grad is None:
                previous_parameter.grad = Parameter(previous_parameter.data.new().resize_(*previous_parameter.data.size()))
            previous_parameter.grad.data.copy_(new_parameter.grad.data)

        # Unscale the gradient
        for previous_parameter in previous_parameters:
            previous_parameter.grad.data /= config.SCALE_FACTOR

        # clip_grad_norm_ mitigates exploding gradients in RNNs as discussed in the final report
        clip_grad_norm_(model.parameters(), config.CLIP_FACTOR)
        optimiser.step()

        # Copy the parameters back into the model
        model_params = list(model.parameters())
        for i in range(len(previous_parameters)):
            model_params[i].data.copy_(previous_parameters[i].data)

    return loss_sum / config.TRAINING_STEPS

"""
Set model to evaluation mode and feed in validation batches
"""
def validation_epoch(model, validation_batch_generator, criterion, epoch):

    p = trange(config.VALIDATION_STEPS, desc='V' + str(epoch))
    loss_sum = 0
    for _ in p:
    
        model.eval()
        midi, moods = validation_batch_generator()
        loss = input_batch(model, midi, moods, criterion, validating=True)
        loss_sum += loss.data.item()
        p.set_postfix(loss=loss.data.item())

    return loss_sum / config.VALIDATION_STEPS


"""
Trains the model on a single batch of sequence by creating inputs and targets for the model to make note predictions with respect to; assess with criterion
"""
def input_batch(model, midi, moods, criterion, validating=False):

    # Declare all variables by converting from the passed tensors
    # One hot the inputs
    inputs = (torch.FloatTensor(midi[:, :-1].size(0), midi[:, :-1].size(1), config.FULL_RANGE)
        .zero_()
        .scatter_(2, midi[:, :-1].unsqueeze(2), 1.0))
    targets = midi[:, 1:]
    if torch.cuda.is_available():
        moods = moods.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()

    # Input data to the model and then calculate the corresponding loss values using CrossEntropyLoss.
    if validating:
        with torch.no_grad():
            outputs, _ = model(inputs, moods, None)
            loss = criterion(outputs.view(-1, config.FULL_RANGE).float(), targets.contiguous().view(-1))
    else:
        outputs, _ = model(inputs, moods, None)
        # print(outputs.view(-1, config.FULL_RANGE).float()[0][targets.contiguous().view(-1)[0]])
        # print(outputs.view(-1, config.FULL_RANGE).float())
        # print(targets.contiguous().view(-1))
        loss = criterion(outputs.view(-1, config.FULL_RANGE).float(), targets.contiguous().view(-1))

    return loss


"""
Saves arrays of loss values and plots a graph each n_epochs of these values
"""
def save_progress(training_losses, validation_losses, name):

    np.save('training_losses.npy', np.array(training_losses))
    np.save('validation_losses.npy', np.array(validation_losses))
    plt.clf()
    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.savefig(config.GRAPHS_DIR + '/' + name)
