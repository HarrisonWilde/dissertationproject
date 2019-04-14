import os, math, argparse, random
import numpy as np
from datetime import datetime
from tqdm import tqdm
try:
    import matplotlib
    if os.environ.get('DISPLAY','') == '':
        print('no display found. Using non-interactive Agg backend')
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except:
    print("matplotlib doesn't work on Mac OS")

start = 0
finish = 160

grutrain_losses = np.load('Pyramid GRU Classical/train_losses.npy').tolist()[start:finish]
gruval_losses = np.load('Pyramid GRU Classical/val_losses.npy').tolist()[start:finish]
lstmtrain_losses = np.load('Pyramid LSTM Classical/train_losses.npy').tolist()[start:finish]
lstmval_losses = np.load('Pyramid LSTM Classical/val_losses.npy').tolist()[start:finish]
gru768train_losses = np.load('512 GRU Classical/train_losses.npy').tolist()[start:finish]
gru768val_losses = np.load('512 GRU Classical/val_losses.npy').tolist()[start:finish]
lstm768train_losses = np.load('512 LSTM Classical/train_losses.npy').tolist()[start:finish]
lstm768val_losses = np.load('512 LSTM Classical/val_losses.npy').tolist()[start:finish]

plt.clf()
plt.figure(figsize=(14,7))
plt.plot(list(range(start, finish)), grutrain_losses, label = "Pyramidal GRU Train", color='#B6B6B6')
plt.plot(list(range(start, finish)), gruval_losses, label = "Pyramidal GRU Validation", ls = '--', color='#B6B6B6')
plt.plot(list(range(start, finish)), lstmtrain_losses, label = "Pyramidal LSTM Train", color='#D6B656')
plt.plot(list(range(start, finish)), lstmval_losses, label = "Pyramidal LSTM Validation", ls = '--', color='#D6B656')
plt.plot(list(range(start, finish)), gru768train_losses, label = "768 GRU Train", color='#809DC8')
plt.plot(list(range(start, finish)), gru768val_losses, label = "768 GRU Validation", ls = '--', color='#809DC8')
plt.plot(list(range(start, finish)), lstm768train_losses, label = "768 LSTM Train", color='#B85551')
plt.plot(list(range(start, finish)), lstm768val_losses, label = "768 LSTM Validation", ls = '--', color='#B85551')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.savefig('loss.png', dpi=350)
