import os
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

import config
from conversion import load_midi


"""
Function to load all MIDI and mood data into lists and return to be used in training and validation, also calculates statistics about the training data
"""
def load_data(split, sequence_length):

    midi_data = []
    mood_data = []
    midi_stats = 0
    mood_stats = [0, 0, 0, 0, 0, 0]
    missing_mood_count = 0
    missed_count = 0
    note_counts = [0] * 128

    for filename in tqdm(os.listdir(config.MIDI_DIR)):

        if filename.endswith(('.mid', '.midi')):
            
            midi = load_midi(os.path.join(config.MIDI_DIR, filename))

            # Count occurrences of each note
            for event in midi:
                if 0 <= event <= 127:
                    note_counts[event] += 1

            if len(midi) >= sequence_length + 10:
                midi_data.append(torch.from_numpy(midi))
                midi_stats += len(midi)
            
            else:
                missed_count += 1
                continue

            if filename.split()[0] in ('16384', '32768'):
                filename = filename[6:]

            try:
                mood = np.load(os.path.join(config.MOOD_DIR, os.path.splitext(filename)[0] + '.npy'), encoding='latin1', allow_pickle=True).item()
                mood_data.append(torch.FloatTensor([
                    1 if mood['valence_sad_ratio'] >= 50 else 0, 
                    1 if mood['valence_neutral_ratio'] >= 50 else 0, 
                    1 if mood['valence_happy_ratio'] >= 50 else 0, 
                    1 if mood['arousal_relaxing_ratio'] >= 50 else 0, 
                    1 if mood['arousal_mid_ratio'] >= 50 else 0, 
                    1 if mood['arousal_intense_ratio'] >= 50 else 0]))
                
                mood_stats[0] += mood['valence_sad_ratio']
                mood_stats[1] += mood['valence_neutral_ratio']
                mood_stats[2] += mood['valence_happy_ratio']
                mood_stats[3] += mood['arousal_relaxing_ratio']
                mood_stats[4] += mood['arousal_mid_ratio']
                mood_stats[5] += mood['arousal_intense_ratio']
            except:
                missing_mood_count += 1
                mood_data.append(torch.FloatTensor([0,0,0,0,0,0]))
    
    print()
    print('Loaded {} MIDI files with an average of {} events per file. Ignored {} files because they were too short. There were {} missing mood files.'.format(len(midi_data), midi_stats / len(midi_data), missed_count, missing_mood_count))
    print()
    print('Note counts: ' + str(note_counts))
    print()
    print('Average mood stats are...')
    print('Valence Sad Ratio: ' + str(mood_stats[0] / len(midi_data)))
    print('Valence Neutral Ratio: ' + str(mood_stats[1] / len(midi_data)))
    print('Valence Happy Ratio: ' + str(mood_stats[2] / len(midi_data)))
    print('Arousal Relaxing Ratio: ' + str(mood_stats[3] / len(midi_data)))
    print('Arousal Mid Ratio: ' + str(mood_stats[4] / len(midi_data)))
    print('Arousal Intense Ratio: ' + str(mood_stats[5] / len(midi_data)))
    print()

    if len(midi_data) == 0:
        raise 'No training data loaded.'

    # Generate a random train:validation split of mood and midi_data data
    indices = np.arange(len(midi_data))
    np.random.shuffle(indices)

    split = int((len(midi_data) * split) + .5)
    train_indices = indices[:-split]
    val_indices = indices[-split:]

    training_midi = [midi_data[i] for i in train_indices]
    training_mood = [mood_data[i] for i in train_indices]
    validation_midi = [midi_data[i] for i in val_indices]
    validation_mood = [mood_data[i] for i in val_indices]

    print('Training Sequences:', len(training_midi), len(training_mood))
    print('Validation Sequences:', len(validation_midi), len(validation_mood))


    return (training_midi, training_mood), (validation_midi, validation_mood)


"""
Creates batches of dimension [batch_size x sequence_length x D_input].
"""
def batch_generator(sampler, batch_size, sequence_length):

    def batch():

        batch = [sampler(sequence_length + 1) for i in range(batch_size)]
        return [torch.stack(x) for x in zip(*batch)]

    return batch 


"""
Generates sampled sequence of MIDI data for training.
"""
def sampler(data, transpose):

    midi_data, mood_data = data

    def sample(sequence_length):

        # Pick a random midi sequence from the data (alongside the corresponding mood) and a random starting index within that sequence to subset into something of length sequence_length
        seq_id = np.random.randint(len(midi_data))
        midi = midi_data[seq_id]
        mood = mood_data[seq_id]
        start_index = np.random.randint(len(midi) - int(sequence_length + 5))
        midi = midi[start_index:start_index + sequence_length]

        if transpose:
            midi = random_tranposition(midi)

        return midi, mood

    return sample


"""
Transposes a midi sequence by some number of semitones, drawn from a Gaussian distribution with variance 3.
"""
def random_tranposition(midi):

    transposed_midi = []
    transposition = int(np.random.normal(0.5, 3))

    # Perform transposition (ensure only notes are moved by the transposition amount, not velocities and times)
    for event in midi:
        if event < config.TIME_OFFSET:
            if (transposition == 0) or (event + transposition >= config.TIME_OFFSET) or (event + transposition < 0):
                return midi
            else:
                transposed_midi.append(event + transposition)
        else:
            transposed_midi.append(event)

    return torch.LongTensor(transposed_midi)