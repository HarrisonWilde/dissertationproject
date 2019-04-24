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


'''
colours = (['#809DC8'] * 128)
colours[60] = '#D6B656'

note_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 540, 1330, 1002, 3680, 5532, 5936, 7074, 8904, 19344, 22144, 31544, 35860, 46984, 45196, 46902, 73452, 62192, 71482, 74594, 81314, 100330, 84892, 135374, 122252, 142070, 139940, 138262, 195594, 158598, 223530, 202422, 220992, 252978, 211798, 311280, 267676, 308926, 307778, 282668, 356892, 298758, 376118, 322842, 345390, 360452, 286316, 374410, 305172, 345202, 302888, 289204, 336786, 276094, 322710, 266342, 269822, 261448, 204622, 245176, 194008, 194164, 160584, 142928, 147384, 115182, 118116, 101140, 91952, 80462, 61476, 59902, 53446, 44504, 39382, 29552, 24004, 25592, 17602, 20542, 14978, 8676, 4176, 2662, 1192, 764, 134, 72, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
total = np.sum(note_counts)
note_percentages = [(x / total) * 100 for x in note_counts]
plt.clf()
plt.figure(figsize=(14,7))
plt.bar(list(range(0,128)), note_percentages, color=colours)
plt.xlabel('MIDI Note Value')
plt.ylabel('Percentage of the Total Number of Notes Present in the Dataset')
axes = plt.gca()
axes.set_ylim([0,3.5])
plt.savefig('classicnotedistrib.png', dpi=350)

new_note_counts = [0] + [int(note_counts[i] + np.random.normal(np.mean(note_counts[i-1:i+1]) - note_counts[i], note_counts[i] * (1/12))) for i in range(1, len(note_counts) - 1)] + [0]
new_total = np.sum(new_note_counts)
new_note_percentages = [(x / new_total) * 100 for x in new_note_counts]
plt.clf()
plt.figure(figsize=(14,7))
plt.bar(list(range(0,128)), new_note_percentages, color=colours)
plt.xlabel('MIDI Note Value')
plt.ylabel('Percentage of the Total Number of Notes Present in the Dataset')
axes = plt.gca()
axes.set_ylim([0,3.5])
plt.savefig('classicnotegendistrib.png', dpi=350)

note_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 206, 234, 68, 146, 270, 126, 140, 556, 758, 11372, 14948, 12008, 14748, 14278, 13552, 17344, 15940, 19462, 19198, 19394, 24328, 36058, 28762, 24012, 27966, 24546, 22430, 26520, 26220, 34140, 27650, 30774, 33564, 38338, 43124, 39868, 57240, 59564, 57874, 53072, 58908, 81094, 49356, 59258, 59168, 47640, 56648, 50676, 72600, 40566, 38880, 41102, 34934, 41044, 24338, 33978, 26004, 22328, 23592, 16886, 21890, 15510, 17832, 16542, 12186, 16426, 10242, 13274, 9726, 9176, 13186, 8142, 8148, 5476, 6554, 5354, 6220, 5738, 3214, 4088, 2730, 2880, 2804, 1136, 840, 556, 138, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
total = np.sum(note_counts)
note_percentages = [(x / total) * 100 for x in note_counts]
plt.clf()
plt.figure(figsize=(14,7))
plt.bar(list(range(0,128)), note_percentages, color=colours)
plt.xlabel('MIDI Note Value')
plt.ylabel('Percentage of the Total Number of Notes Present in the Dataset')
axes = plt.gca()
axes.set_ylim([0,4.25])
plt.savefig('ambientnotedistrib.png', dpi=350)

new_note_counts = [0] + [int(note_counts[i] + np.random.normal(np.mean(note_counts[i-1:i+1]) - note_counts[i], note_counts[i] * (1/8))) for i in range(1, len(note_counts) - 1)] + [0]
new_total = np.sum(new_note_counts)
new_note_percentages = [(x / new_total) * 100 for x in new_note_counts]
plt.clf()
plt.figure(figsize=(14,7))
plt.bar(list(range(0,128)), new_note_percentages, color=colours)
plt.xlabel('MIDI Note Value')
plt.ylabel('Percentage of the Total Number of Notes Present in the Dataset')
axes = plt.gca()
axes.set_ylim([0,4.25])
plt.savefig('ambientnotegendistrib.png', dpi=350)
'''
'''