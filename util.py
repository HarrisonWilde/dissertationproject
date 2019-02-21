import numpy as np
import tensorflow as tf
import math

from constants import *
from midi_util import *

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

