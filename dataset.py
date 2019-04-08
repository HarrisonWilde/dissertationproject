"""
Preprocesses MIDI files
"""
import math, random, itertools, torch, os
import numpy as np
from tqdm import tqdm

from conversion import load_midi
from util import *
import config

def load():
    """
    Loads all music styles into a list of compositions
    """
    seqs = []
    moods = []
    seqs_sum = 0
    mood_stats = [0, 0, 0, 0, 0, 0]

    for filename in tqdm(os.listdir(config.MIDI_DIR)):

        if filename.endswith(('.mid', '.midi')):
            try:
                # Pad the sequence by an empty event
                seq = load_midi(os.path.join(config.MIDI_DIR, filename))

                if len(seq) >= config.SEQ_LEN * 1.5:
                    seqs.append(torch.from_numpy(seq).long())
                    seqs_sum += len(seq)
                else:
                    print('Ignoring {} because it is too short {}.'.format(filename, len(seq)))
                    pass

            except Exception as e:
                print('Unable to load ' + filename, e)

            if filename.split()[0] in ('16384', '32768'):
                filename = filename[6:]
            try:
                mood = np.load(os.path.join(config.MOOD_DIR, os.path.splitext(filename)[0] + '.npy'), encoding='latin1').item()
                moods.append(torch.FloatTensor([
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
                print("no mood file found")
                moods.append(torch.FloatTensor([0,0,0,0,0,0]))
        
    print('Loaded {} MIDI files with average event count {}'.format(len(seqs), seqs_sum / len(seqs)))
    print()
    print('Average mood stats are...')
    print('Valence Sad Ratio: ' + str(mood_stats[0] / len(seqs)))
    print('Valence Neutral Ratio: ' + str(mood_stats[1] / len(seqs)))
    print('Valence Happy Ratio: ' + str(mood_stats[2] / len(seqs)))
    print('Arousal Relaxing Ratio: ' + str(mood_stats[3] / len(seqs)))
    print('Arousal Mid Ratio: ' + str(mood_stats[4] / len(seqs)))
    print('Arousal Intense Ratio: ' + str(mood_stats[5] / len(seqs)))

    return seqs, moods

def validation_split(data, split=0.2):
    """
    Splits the data iteration list into training and validation indices
    """
    seqs, moods = data

    # Shuffle sequences randomly
    r = list(range(len(seqs)))
    random.shuffle(r)

    num_val = int(math.ceil(len(r) * split))
    train_indicies = r[:-num_val]
    val_indicies = r[-num_val:]

    assert len(val_indicies) == num_val
    assert len(train_indicies) == len(r) - num_val

    train_seqs = [seqs[i] for i in train_indicies]
    val_seqs = [seqs[i] for i in val_indicies]

    train_mood_tags = [moods[i] for i in train_indicies]
    val_mood_tags = [moods[i] for i in val_indicies]
    
    return (train_seqs, train_mood_tags), (val_seqs, val_mood_tags)

def sampler(data):
    """
    Generates sequences of data.
    """
    seqs, moods = data

    if len(seqs) == 0:
        raise 'Insufficient training data.'

    def sample(seq_len):
        # Pick random sequence
        seq_id = random.randint(0, len(seqs) - 1)
        seq = seqs[seq_id]
        # Pick random start index
        start_index = random.randint(0, len(seq) - 1 - int(seq_len * 1.5))
        seq = seq[start_index:]
        # Apply random augmentations
        seq = augment(seq)
        # Take first N elements. After augmentation seq len changes.
        seq = itertools.islice(seq, seq_len)
        seq = gen_to_tensor(seq)
        assert seq.size() == (seq_len,), seq.size()

        return (seq, moods[seq_id])
    return sample

def batcher(sampler, batch_size, seq_len=config.SEQ_LEN):
    """
    Bundles samples into batches
    """
    def batch():
        batch = [sampler(seq_len) for i in range(batch_size)]
        return [torch.stack(x) for x in zip(*batch)]
    return batch 

# def stretch_sequence(sequence, stretch_scale):
#     """ Iterate through sequence and stretch each time shift event by a factor """
#     # Accumulated time in seconds
#     time_sum = 0
#     seq_len = 0
#     for i, evt in enumerate(sequence):
#         if evt >= TIME_OFFSET and evt < VELOCITY_OFFSET:
#             # This is a time shift event
#             # Convert time event to number of seconds
#             # Then, accumulate the time
#             time_sum += convert_time_evt_to_sec(evt)
#         else:
#             if i > 0:
#                 # Once there is a non time shift event, we take the
#                 # buffered time and add it with time stretch applied.
#                 for x in seconds_to_events(time_sum * stretch_scale):
#                     yield x
#                 # Reset tracking variables
#                 time_sum = 0
#             seq_len += 1
#             yield evt

#     # Edge case where last events are time shift events
#     if time_sum > 0:
#         for x in seconds_to_events(time_sum * stretch_scale):
#             seq_len += 1
#             yield x

#     # Pad sequence with empty events if seq len not enough
#     if seq_len < config.SEQ_LEN:
#         for x in range(SEQ_LEN - seq_len):
#             yield config.TIME_OFFSET
            
def transpose(sequence):
    """ A generator that represents the sequence. """
    # Transpose by 4 semitones at most
    transpose = random.randint(-4, 4)

    if transpose == 0:
        return sequence

    # Perform transposition (consider only notes)
    return (evt + transpose if evt < config.TIME_OFFSET else evt for evt in sequence)

def augment(sequence):
    """
    Takes a sequence of events and randomly perform augmentations.
    """
    sequence = transpose(sequence)
    # sequence = stretch_sequence(sequence, random.uniform(1.0, 1.25))
    return sequence
