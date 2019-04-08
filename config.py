# Representation details, MIDI characteristics to capture
MIDI_VELOCITY = 128
NOTE_RANGE = 128
# Use bins to quantise values
TIME_BINS = 32
VELOCITY_BINS = 32

TIME_OFFSET = NOTE_RANGE
VELOCITY_OFFSET = TIME_OFFSET + TIME_BINS
FULL_RANGE = VELOCITY_OFFSET + VELOCITY_BINS

# Time representation details, MIDI tick
TICK_EXP = 1.14
TICK_MUL = 1
# The number of ticks represented in each bin
TICK_BINS = [int(TICK_EXP ** x + TICK_MUL * x) for x in range(TIME_BINS)]
TICKS_PER_SEC = 100

# Model train parameters
SEQ_LEN = 1025
GRADIENT_CLIP = 10
SCALE_FACTOR = 2 ** 10

# The number of train generator cycles per sequence
TRAIN_CYCLES = 250
VAL_CYCLES = int(TRAIN_CYCLES * 0.2)

# Directories
AUDIO_DIR = '../Audio2.nosync'
CACHE_DIR = 'out/cache'
GRAPHS_DIR = 'out/graphs'
MIDI_DIR = 'data/midi'
MODELS_DIR = 'out/models'
MOOD_DIR = 'data/mood'
SAMPLES_DIR = 'out/samples'
