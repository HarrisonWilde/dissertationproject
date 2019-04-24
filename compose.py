import argparse
import heapq
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange

import config
from conversion import *


"""
Represents a music generation sequence
"""
class Composer():

    def __init__(self, model, mood=None, default_temp=0.9, beam_size=1):
        
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

    def step(self):
        """
        Generates the next set of beams
        """
        # Create variables
        mood = Variable(torch.from_numpy(self.mood).float())
        if torch.cuda.is_available():
            mood = mood.cuda()
        mood = mood.unsqueeze(0)

        new_beam = []
        sum_seq_prob = 0

        # Iterate through the beam
        for prev_prob, evts, state in self.beam:
            if len(evts) > 0:
                temp = np.zeros(config.FULL_RANGE)
                temp[evts[-1]] = 1
                prev_event = Variable(torch.from_numpy(temp).float())
                if torch.cuda.is_available():
                    prev_event = prev_event.cuda()
                prev_event = prev_event.unsqueeze(0)
            else:
                prev_event = Variable(torch.zeros((1, config.FULL_RANGE)))
                if torch.cuda.is_available():
                    prev_event = prev_event.cuda()

            prev_event = prev_event.unsqueeze(1)
            probs, new_state = self.model.compose(prev_event, mood, state, temperature=self.temperature)
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
        self.step_count += 1

    """
    Compose a piece and export it to MIDI
    """
def compose(self, name='output', sequence_length=5000):
    self.model.eval()

        for _ in trange(sequence_length):
            self.step()

        chosen_composition = np.array(max(self.beam, key=lambda x: x[0])[1])
        save_midi(name, chosen_composition)
