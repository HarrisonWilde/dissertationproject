"""
Preprocesses MIDI files
"""
import numpy as np
import math
import random
from joblib import Parallel, delayed
import multiprocessing

from constants import *
from midi_util import load_midi
from util import *

def compute_beat(beat, notes_in_bar):
    return one_hot(beat % notes_in_bar, notes_in_bar)

def compute_completion(beat, len_melody):
    return np.array([beat / len_melody])

def load_all(batch_size, time_steps, notes_per_bar, path):
    """
    Loads all MIDI files from the given path into matrices.
    (For use with Keras model)
    """
    def stagger(data):
        dataX, dataY = [], []
        # Buffer training for first event
        data = ([np.zeros_like(data[0])] * time_steps) + list(data)

        # Chop a sequence into measures
        for i in range(0, len(data) - time_steps, notes_per_bar):
            dataX.append(data[i:i + time_steps])
            dataY.append(data[i + 1:(i + time_steps + 1)])
        return dataX, dataY

    note_data = []
    beat_data = []
    note_target = []
    filenames = []

    for filename in os.listdir(path):
        if filename.endswith('.mid'):
            filenames.append(os.path.join(path, filename))

    # Parallel process all files into a list of music sequences
    seqs = Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading')(delayed(load_midi)(file) for file in filenames)

    for seq in seqs:
        if len(seq) >= time_steps:
            # Clamp MIDI to note range
            seq = clamp_midi(seq)
            # Create training data and labels
            train_data, label_data = stagger(seq)
            note_data += train_data
            note_target += label_data

            beats = [compute_beat(i, notes_per_bar) for i in range(len(seq))]
            beat_data += stagger(beats)[0]

    note_data = np.array(note_data)
    beat_data = np.array(beat_data)
    note_target = np.array(note_target)
    return [note_data, note_target, beat_data], [note_target]

def clamp_midi(sequence):
    """
    Removes all notes outside of the defined MIN and MAX notes
    """
    return sequence[:, MIN_NOTE:MAX_NOTE, :]

def unclamp_midi(sequence):
    """
    Restore clamped MIDI sequence back to MIDI note values
    """
    return np.pad(sequence, ((0, 0), (MIN_NOTE, 0), (0, 0)), 'constant')
