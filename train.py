import midi, os


import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.callbacks import EarlyStopping, TensorBoard

from constants import *
from dataset import *
from generate import *
from model import *

def train(models, path='data', batch_size=10, notes_per_bar=16, bars=8):
    
    print('Loading data')
    train_data, train_labels = load_all(batch_size, notes_per_bar * bars, notes_per_bar, path)

    print train_data
    print train_labels

    cbs = [
        ModelCheckpoint(MODEL_FILE, monitor='loss', save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor='loss', patience=5),
        TensorBoard(log_dir='out/logs', histogram_freq=0)
    ]

    print('Training')
    # models[0].fit(train_data, train_labels, epochs=1000, callbacks=cbs, batch_size=batch_size)
