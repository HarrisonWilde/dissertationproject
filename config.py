"""
Directories
"""
AUDIO_DIR = '../Audio2.nosync'
COMPOSITIONS_DIR = 'output/compositions'
GRAPHS_DIR = 'output/graphs'
MODELS_DIR = 'output/models'
MIDI_DIR = 'data/midi'
MOOD_DIR = 'data/mood'


"""
Fundamental characteristics of MIDI
"""
VELOCITY_RANGE = 128
NOTE_RANGE = 128


"""
Bins to round time and velocity values, effectively a x4 downscale in resolution for velocity, time is more complicated
"""
VELOCITY_BINS = 32
NUM_TIME_BINS = 32

# The number of ticks represented in each bin
TIME_BINS = [int(1.2 ** i + i) for i in range(NUM_TIME_BINS)]
TICKS_PER_SEC = 100


"""
Representation details, to correspond with D_input in the report
"""
TIME_OFFSET = NOTE_RANGE
VELOCITY_OFFSET = TIME_OFFSET + NUM_TIME_BINS
FULL_RANGE = VELOCITY_OFFSET + VELOCITY_BINS
MOOD_DIMENSION = 6


"""
Model training parameters
"""
TRAINING_STEPS = 250
VALIDATION_STEPS = 75