import argparse
import tensorflow as tf

from datetime import datetime
from train import train
from generate import generate, write_file
from model import build

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

def main():

    p = argparse.ArgumentParser(description='WildeNet compositional engine')
    p.add_argument('-tl', '--time_layers', default=[256,256], type=int, nargs='+', 
        help='List of LSTM layer sizes for the time model, in the format x y z ...')
    p.add_argument('-hl', '--harm_layers', default=[128,64], type=int, nargs='+', 
        help='List of LSTM layer sizes for the time model, in the format x y z ...')
    p.add_argument('-t', '--train', action='store_true',
        help='Flag to train model, this occurs by default and is included for completeness')
    p.add_argument('-p', '--path', default='data', type=str, 
        help='Path to MIDI files for use in training the compositional engine DEFAULT IS ./data')
    p.add_argument('-bs', '--batch_size', default=10, type=int, 
        help='Training batch size to feed into the models DEFAULT IS 10')
    p.add_argument('-b', '--bars', default=8, type=int, 
        help='Number of bars per batch of training data, this flag is for training a model, does not effect composition length DEFAULT IS 8')
    p.add_argument('-bpb', '--beats_per_bar', default=4, type=int, 
        help='Number of beats per training bar DEFAULT IS 4')
    p.add_argument('-npb', '--notes_per_beat', default=4, type=int, 
        help='Number of notes per training beat DEFAULT IS 4')
    p.add_argument('-g', '--generate', action='store_true',
        help='Flag to use the generative model to compose a piece')
    p.add_argument('-n', '--name', default='output ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), type=str,
        help='Name for resulting composition DEFAULT IS "output <timestamp>"')
    p.add_argument('-nb', '--num_bars', default=64, type=int, 
        help='Number of bars of music to generate when -g flag is set, the length of the resulting composition depends on this argument DEFAULT IS 64')
    p.add_argument('-nc', '--no_cache', action='store_true',
        help='Add this flag to not attempt to use any previously trained model files etc.')
    args = p.parse_args()

    notes_per_bar = args.notes_per_beat * args.beats_per_bar
    models = build(args.time_layers, args.harm_layers, notes_per_bar * args.bars, notes_per_bar)
    
    if not args.no_cache:
        try:
            models[0].load_weights(MODEL_FILE)
            print('Loaded model from file.')
        except:
            print('Unable to load model from file.')
    
    if args.generate:
        write_file(args.name, generate(models, args.num_bars))
    else:
        models[0].summary()
        train(models, args.path, args.batch_size, notes_per_bar, args.bars)

if __name__ == '__main__':
    main()