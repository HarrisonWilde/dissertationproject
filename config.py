### MIDI Parameters ###
MIDI_VELOCITY = 128
# Number of possible notes
NUM_NOTES = 128
# Number of time shift quantizations
TIME_QUANTIZATION = 32
# Exponential representation of time shifts
TICK_EXP = 1.14
TICK_MUL = 1
# The number of ticks represented in each bin
TICK_BINS = [int(TICK_EXP ** x + TICK_MUL * x) for x in range(TIME_QUANTIZATION)]
# Ticks per second
TICKS_PER_SEC = 100
# Number of velocity buns
VEL_QUANTIZATION = 32

NOTE_ON_OFFSET = 0
TIME_OFFSET = NOTE_ON_OFFSET + NUM_NOTES
VEL_OFFSET = TIME_OFFSET + TIME_QUANTIZATION
NUM_ACTIONS = VEL_OFFSET + VEL_QUANTIZATION

# Trainin Parameters
SEQ_LEN = 1024 + 1
GRADIENT_CLIP = 10
SCALE_FACTOR = 2 ** 10
# The number of train generator cycles per sequence
TRAIN_CYCLES = 250
VAL_CYCLES = int(TRAIN_CYCLES * 0.2)

# Mood
NUM_MOODS = 6

# Paths
AUDIO_DIR = '../Audio2.nosync'
MOOD_DIR = 'data/mood'
MIDI_DIR = 'data/midi'
OUT_DIR = 'out'
CACHE_DIR = 'out/cache'
SAMPLES_DIR = 'out/samples'

settings = {
    'force_cpu': False
}
