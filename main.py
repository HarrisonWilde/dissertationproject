import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from datetime import datetime

import config
from data import *
from compose import compose
from conversion import save_midi
from model import Model
from train import train


"""
Parser provides options to a user at runtime, can choose between training and generating with -g and also set a lot of parameters for both processes
"""
parser = argparse.ArgumentParser(description='Interact with the RDLSTM / RDGRU deep learning architecture for musical composition.')
parser.add_argument('-m', '--model', help='Path to an existing model file (assumed to be in "output/models/".')
parser.add_argument('-bs', '--batch-size', default=64, type=int, help='Set size of each batch for training.')
parser.add_argument('-s', '--split', default=0.25, type=float, help='Train / Validation split to apply to the training data.')
parser.add_argument('-t', '--transpose', default=True, action='store_true', help='Apply random transpositions to the training data with this option.')
parser.add_argument('-o', '--optimiser', default='Adam', type=str, help='Optimiser to use, choose from any torch.optim optimisers.')
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float, help='Learning rate during training.')
parser.add_argument('-g', '--generate', default=False, action='store_true', help='Flag to enable generation with an existing model rather than the training of a new / existing model.')
parser.add_argument('-n', '--name', default='output ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), help='Name of the output composition file(s).')
parser.add_argument('-mi', '--mood-input', default=None, type=int, nargs='+', help='Specify the mood characteristics to aim for, default will generate one piece for each possible mood input.')
parser.add_argument('-l', '--length', default=5000, type=int, help='Length of generated piece(s).')
parser.add_argument('-te', '--temperature', default=0.8, type=float, help='Set the generation temperature value, default is 0.8.')
parser.add_argument('-c', '--n-candidates', default=1, type=int, help='Set the number of candidates to consider during generation steps, default is 1.')
args = parser.parse_args()

print('Loading Model...')
print()
model = Model()

# Utilise fp16 to increase performance during training
if torch.cuda.is_available():

    model.cuda()
    fwd = model.forward
    model.forward = lambda x, mood, states: fwd(x.half(), mood.half(), states)
    model.half()

# Try to load parameters from a passed model location
if args.model:
    model.load_state_dict(torch.load(config.MODELS_DIR + '/' + args.path))
    print('Loaded previous model parameters.')
else:
    print('No / invalid model path specified; new model created.')

if torch.cuda.is_available():
    print('Using available GPU.')
else:
    print('No GPU available, falling back on CPU.')

"""
Either generate using an existing model or train an existing / new model, based on args.generate
"""
if args.generate:

    # Make relevant directories
    os.makedirs(config.COMPOSITIONS_DIR, exist_ok=True)

    if args.mood:
        moods = [np.array(args.mood)]
    else:
        moods = [np.array(i) for i in itertools.product([0,1], repeat=6)]
    
    print()
    print('Composing pieces...')

    for i in trange(len(moods)):

        print('Composition: {}'.format(args.name + ' ' + str(moods[i])))
        composition = compose(model, moods[i], args.temperature, args.n_candidates, args.length)
        save_midi(args.name + ' ' + str(moods[i]), composition)

else:
    
    # Make relevant directories
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.GRAPHS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Set up optimiser for training
    param_copy = [p.clone().type(torch.FloatTensor).detach() for p in model.parameters()]
    for p in param_copy:
        p.requires_grad = True
    optimiser = getattr(optim, args.optimiser)(param_copy, lr=args.learning_rate, eps=1e-5)
    model.param_copy = param_copy

    print()
    print('Loading data...')
    training_data, validation_data = load_data(split=args.split)
    
    training_batch_generator = batch_generator(sampler(training_data, args.transpose), args.batch_size, config.SEQUENCE_LENGTH)
    validation_batch_generator = batch_generator(sampler(validation_data, args.transpose), args.batch_size, config.SEQUENCE_LENGTH)
    
    print()
    print('Training...')
    criterion = CrossEntropyLoss()
    train(model, training_batch_generator, config.TRAINING_STEPS, validation_batch_generator, config.VALIDATION_STEPS, optimiser, criterion)
