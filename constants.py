import os

# MIDI Resolution
DEFAULT_RES = 96
MIDI_MAX_NOTES = 128
MAX_VELOCITY = 127

# Number of octaves supported
NUM_OCTAVES = 6

# Min and max note (in MIDI note number)
MIN_NOTE = 24
MAX_NOTE = MIN_NOTE + NUM_OCTAVES * 12
NUM_NOTES = MAX_NOTE - MIN_NOTE

# Hyper Parameters
OCTAVE_UNITS = 64
NOTE_UNITS = 3

# Move file save location
OUT_DIR = 'out'
MODEL_DIR = os.path.join(OUT_DIR, 'models')
MODEL_FILE = os.path.join(OUT_DIR, 'model.h5')
SAMPLES_DIR = os.path.join(OUT_DIR, 'samples')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
