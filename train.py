import os, math, argparse, random
import numpy as np
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

import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import *
from util import *
from model import WildeNet
from conversion import save_midi

# output_graph = True

# Equivalent to negative log-likelihood
criterion = nn.CrossEntropyLoss()

def plot_loss(training_loss, validation_loss, name):
    # Draw graph
    plt.clf()
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.savefig(config.GRAPHS_DIR + '/' + name)

def train(args, model, train_batcher, train_len, val_batcher, val_len, optimizer, plot=True):
    """
    Trains a model on multiple seq batches by iterating through a generator.
    """
    # Number of training steps per epoch
    epoch = 1
    total_step = 1

    # Keep tracks of all losses in each epoch
    try:
        train_losses = np.load('train_losses.npy').tolist()
        val_losses = np.load('val_losses.npy').tolist()
    except:
        print("Starting with fresh train and validation loss lists")
        train_losses = []
        val_losses = []

    # Epoch loop
    while True:
        # Training
        step = 1
        total_loss = 0

        with tqdm(range(train_len)) as t:
            t.set_description('Epoch {}'.format(epoch))
            
            for _ in t:
                data = train_batcher()
                loss = train_step(model, data, optimizer)

                total_loss += loss
                avg_loss = total_loss / step
                t.set_postfix(loss=avg_loss)

                step += 1
                total_step += 1

        train_losses.append(avg_loss)

        # Validation
        step = 1
        total_loss = 0

        with tqdm(range(val_len)) as t:
            t.set_description('Validation {}'.format(epoch))

            for _ in t:
                data = val_batcher()
                loss = val_step(model, data)
                total_loss += loss
                avg_loss = total_loss / step
                t.set_postfix(loss=avg_loss)
                step += 1
            
        val_losses.append(avg_loss)

        if plot:
            np.save('train_losses.npy', np.array(train_losses))
            np.save('val_losses.npy', np.array(val_losses))
            plot_loss(train_losses, val_losses, 'loss_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.png')

        # Save model
        torch.save(model.state_dict(), config.MODELS_DIR + '/model_' + str(epoch) + '.pt')

        epoch += 1

def train_step(model, data, optimizer):
    """
    Trains the model on a single batch of sequence.
    """
    model.train()

    loss, avg_loss = compute_loss(model, data)
    
    # Scale the loss
    loss = loss * config.SCALE_FACTOR

    # Zero out the gradient
    model.zero_grad()
    loss.backward()
    param_copy = model.param_copy
    set_grad(param_copy, list(model.parameters()))

    # Unscale the gradient
    if config.SCALE_FACTOR != 1:
        for param in param_copy:
            param.grad.data /= config.SCALE_FACTOR

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    # Reference: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
    optimizer.step()

    # Copy the parameters back into the model
    copy_in_params(model, param_copy)
    return avg_loss

def val_step(model, data):
    model.eval()
    return compute_loss(model, data, volatile=True)[1]

def compute_loss(model, data, volatile=False):
    """
    Trains the model on a single batch of sequence.
    """
    # Convert all tensors into variables
    note_seq, moods = data
    
    moods = var(moods)

    # Feed it to the model
    inputs = var(one_hot_seq(note_seq[:, :-1], config.FULL_RANGE))
    # print(inputs)
    # print(inputs[0])
    # print(inputs[0][0])
    # print(len(inputs))
    # print(len(inputs[0]))
    # print(len(inputs[0][0]))
    targets = var(note_seq[:, 1:])
    if volatile:
        with torch.no_grad():
            output, _ = model(inputs, moods, None)

            # Compute the loss.
            # Note that we need to convert this back into a float because it is a large summation.
            # Otherwise, it will result in 0 gradient.
            loss = criterion(output.view(-1, config.FULL_RANGE).float(), targets.contiguous().view(-1))
    else:
        # if output_graph:
        #     import hiddenlayer
        #     graph = hiddenlayer.build_graph(model, (inputs, moods))
        #     graph.save('./graph')
        output, _ = model(inputs, moods, None)
        print(output.view(-1, config.FULL_RANGE).float()[0][targets.contiguous().view(-1)[0]])
        print(targets.contiguous().view(-1))
        loss = criterion(output.view(-1, config.FULL_RANGE).float(), targets.contiguous().view(-1))

    return loss, loss.data.item()

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--path', help='Load existing model?')
    parser.add_argument('--batch-size', default=128, type=int, help='Size of the batch')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--noplot', default=False, action='store_true', help='Do not plot training/loss graphs')
    parser.add_argument('--no-fp16', default=False, action='store_true', help='Disable FP16 training')
    args = parser.parse_args()
    args.fp16 = not args.no_fp16

    print('=== Loading Model ===')
    model = WildeNet()

    if torch.cuda.is_available():
        model.cuda()

        if args.fp16:
            # Wrap forward method
            fwd = model.forward
            model.forward = lambda x, mood, states: fwd(x.half(), mood.half(), states)
            model.half()

    if args.path:
        model.load_state_dict(torch.load(config.MODELS_DIR + '/' + args.path))
        print('Restored model from checkpoint.')

    # Construct optimizer
    param_copy = [param.clone().type(torch.FloatTensor).detach() for param in model.parameters()]
    for param in param_copy:
        param.requires_grad = True
    optimizer = optim.Adam(param_copy, lr=args.lr, eps=1e-4)
    model.param_copy = param_copy

    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

    print('GPU: {}'.format(torch.cuda.is_available()))
    print('Batch Size: {}'.format(args.batch_size))
    print('FP16: {}'.format(args.fp16))
    print('# of Parameters: {}'.format(params))

    print()

    print('=== Dataset ===')
    os.makedirs('out', exist_ok=True)
    print('Loading data...')
    data = load()
    print()
    print('Creating data generators...')
    train_data, val_data = validation_split(data)
    train_batcher = batcher(sampler(train_data), args.batch_size)
    val_batcher = batcher(sampler(val_data), args.batch_size)

    # Checks if training data sounds right.
    # for i, seq in enumerate(train_batcher()[0]):
    #     save_midi('train_seq_{}'.format(i), seq.cpu().numpy())

    print('Training Sequences:', len(train_data[0]), 'Validation Sequences:', len(val_data[0]))
    print()

    print('=== Training ===')
    train(args, model, train_batcher, config.TRAIN_CYCLES, val_batcher, config.VAL_CYCLES, optimizer, plot=not args.noplot)

if __name__ == '__main__':
    main()
