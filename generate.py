import numpy as np
import argparse
import heapq
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange

import config
from conversion import *
from dataset import *
from util import *
from model import WildeNet

class Generation():
    """
    Represents a music generation sequence
    """

    def __init__(self, model, mood=None, default_temp=0.9, beam_size=1, adaptive=False):
        self.model = model

        self.beam_size = beam_size

        self.mood = mood

        # Temperature of generation
        self.default_temp = default_temp
        self.temperature = self.default_temp

        # Model parametrs
        self.beam = [
            (1, tuple(), None)
        ]
        self.avg_seq_prob = 1
        self.step_count = 0
        self.adaptive_temp = adaptive

    def step(self):
        """
        Generates the next set of beams
        """
        # Create variables
        mood = var(to_torch(self.mood)).unsqueeze(0)

        new_beam = []
        sum_seq_prob = 0

        # Iterate through the beam
        for prev_prob, evts, state in self.beam:
            if len(evts) > 0:
                prev_event = var(to_torch(one_hot(evts[-1], config.NUM_ACTIONS))).unsqueeze(0)
            else:
                prev_event = var(torch.zeros((1, config.NUM_ACTIONS)))

            prev_event = prev_event.unsqueeze(1)
            probs, new_state = self.model.generate(prev_event, mood, state, temperature=self.temperature)
            probs = probs.squeeze(1)

            for _ in range(self.beam_size):
                # Sample action
                output = probs.multinomial(num_samples=1).data
                event = output[0, 0]
                
                # Create next beam
                seq_prob = prev_prob * probs.data[0, event]
                # Boost the sequence probability by the average
                new_beam.append((seq_prob / self.avg_seq_prob, evts + (event,), new_state))
                sum_seq_prob += seq_prob

        self.avg_seq_prob = sum_seq_prob / len(new_beam)
        # Find the top most probable sequences
        self.beam = heapq.nlargest(self.beam_size, new_beam, key=lambda x: x[0])

        if self.adaptive_temp and self.step_count > 50:
            r = repetitiveness(self.beam[0][1][-50:])
            if r < 0.1:
                self.temperature = self.default_temp
            else:
                self.temperature += 0.05
        
        self.step_count += 1

    def generate(self, seq_len=1000, show_progress=True):
        self.model.eval()
        r = trange(seq_len) if show_progress else range(seq_len)

        for _ in r:
            self.step()

        best = max(self.beam, key=lambda x: x[0])
        best_seq = best[1]
        return np.array(best_seq)

    def export(self, name='output', seq_len=1000, show_progress=True):
        """
        Export into a MIDI file.
        """
        seq = self.generate(seq_len, show_progress=show_progress)
        save_midi(name, seq)

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--name', default='output ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), help='Name of the output file')
    parser.add_argument('--model', help='Path to model file')
    parser.add_argument('--length', default=5000, type=int, help='Length of generation')
    parser.add_argument('--mood', default=None, type=int, nargs='+', help='Specify the mood characteristics to aim for, default will generate without a particular mood influence.')
    parser.add_argument('--temperature', default=0.9, type=float, help='Temperature of generation')
    parser.add_argument('--beam', default=1, type=int, help='Beam size')
    parser.add_argument('--adaptive', default=False, action='store_true', help='Adaptive temperature')
    args = parser.parse_args()

    mood = []

    if args.mood:
        # Custom mood
        moods = [np.array(args.mood)]
    else:
        # Generate all possible mood
        # mood = np.array([37.25, 49.99, 55.1, 98.36, 36.09, 7.91])
        moods = [np.array(i) for i in itertools.product([0,1], repeat=6)]
    
    print('=== Loading Model ===')
    print('Path: {}'.format(args.model))
    print('Temperature: {}'.format(args.temperature))
    print('Beam: {}'.format(args.beam))
    print('Adaptive Temperature: {}'.format(args.adaptive))
    print('Moods: {}'.format(mood))
    settings['force_cpu'] = True
    
    model = WildeNet()

    if args.model:
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
    else:
        print('WARNING: No model loaded! Please specify model path.')

    print('=== Generating ===')

    for mood in moods:
        print('File: {}'.format(args.name + ' ' + str(mood)))
        generation = Generation(model, mood=mood, default_temp=args.temperature, beam_size=args.beam, adaptive=args.adaptive)
        generation.export(name=args.name + ' ' + str(mood), seq_len=args.length)

if __name__ == '__main__':
    main()
