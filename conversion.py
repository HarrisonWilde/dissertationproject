import math
import mido
import os
import numpy as np

import config


"""
Builds a track from an iterator of passed events
"""
class TrackBuilder():
    def __init__(self, event_seq, tempo=mido.bpm2tempo(120)):
        self.event_seq = event_seq
        
        self.prev_velocity = 0
        self.delta_time = 0
        self.tempo = mido.bpm2tempo(120)
        self.track_tempo = tempo
        
        self.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        evt = next(self.event_seq).item()

        # Interpret event data
        if evt >= config.VELOCITY_OFFSET:
            # A velocity change
            self.prev_velocity = (evt - config.VELOCITY_OFFSET) * (config.VELOCITY_RANGE // config.VELOCITY_BINS)
        elif evt >= config.TIME_OFFSET:
            # Shifting forward in time
            time_bin = evt - config.TIME_OFFSET
            assert time_bin >= 0 and time_bin < config.NUM_TIME_BINS
            seconds = config.TIME_BINS[time_bin] / config.TICKS_PER_SEC
            self.delta_time += int(mido.second2tick(seconds, self.midi.ticks_per_beat, self.tempo))
        elif evt >= 0:
            # Turning a note on (or off if velocity = 0)
            note = evt
            # We can turn a note on twice, indicating a replay
            if self.prev_velocity == 0:
                # Note off
                if note in self.on_notes:
                    # We cannot turn a note off when it was never on
                    self.track.append(mido.Message('note_off', note=note, time=self.delta_time))
                    self.on_notes.remove(note)
                    self.delta_time = 0
            else:
                self.track.append(mido.Message('note_on', note=note, time=self.delta_time, velocity=self.prev_velocity))
                self.on_notes.add(note)
                self.delta_time = 0
        
    def reset(self):
        self.midi = mido.MidiFile()
        self.track = mido.MidiTrack()
        self.track.append(mido.MetaMessage('set_tempo', tempo=self.track_tempo))
        # Tracks on notes
        self.on_notes = set()
    
    def export(self):
        for _ in self:
            pass
        self.midi.tracks.append(self.track)
        midi_out = self.midi
        self.reset()
        return midi_out


"""
Returns an event sequence representation of the MIDI file associated with the passed path
"""
def load_midi(filename):

    cached_midi = os.path.join(config.CACHE_DIR, filename + '.npy')

    if os.path.isfile(cached_midi):
        event_seq = np.load(cached_midi)
    
    else:
        midi = mido.MidiFile(filename)
        event_seq = rep_from_midi(tpb, mido.merge_tracks(midi.tracks))

        # Cache the event sequence to avoid having to wait next time
        os.makedirs(os.path.dirname(cached_midi), exist_ok=True)
        np.save(cached_midi, event_seq)
    
    return event_seq


"""
Saves the passed event sequence representation as a MIDI file with name filename
"""
def save_midi(filename, event_seq):

    os.makedirs(config.COMPOSITIONS_DIR, exist_ok=True)
    midi = TrackBuilder(iter(event_seq)).export()
    midi.save(config.COMPOSITIONS_DIR + '/' + filename + '.mid')


"""
Convert from MIDI to the representation defined in the report
"""
def rep_from_midi(tpb, track):

    events = []
    tempo = 120
    prev_velocity = None
    
    for event in track:
        
        # Check meta event type to see if tempo can be set, else ignore
        if event.is_meta:
            if event.type == 'set_tempo':
                tempo = event.tempo

        # Calculate time events to append according to the time that has passed since the previous event
        if event.time != 0:
            events += list(bin_time_to_events(mido.tick2second(event.time, tpb, tempo)))

        # Ignore events that do not pertain to a note being played / released
        if event.type != 'note_on' and event.type != 'note_off':
            continue

        # Quantise velocity of notes
        if event.type == 'note_on':
            velocity = event.velocity // (config.VELOCITY_RANGE // config.VELOCITY_BINS)
        else:
            velocity = 0
        
        if prev_velocity != velocity:
            events.append(config.VELOCITY_OFFSET + velocity)
            prev_velocity = velocity

        events.append(event.note)

    return np.array(events)


"""
Applies binning to the passed length of time, returning multiple events if maximum bin size is reached, seconds are converted to events
"""
def bin_time_to_events(seconds):

    ticks = round(seconds * config.TICKS_PER_SEC)

    while ticks > 0:

        # Use current number of ticks to find largest possible bin
        for i, bin_ticks in enumerate(config.TIME_BINS):
            if ticks >= bin_ticks:
                time_bin = i

        # Calculate remaining ticks for next iteration
        ticks -= config.TIME_BINS[time_bin]
        
        # Yield a time event to append to the sequence corresponding to the calculated bin
        yield config.TIME_OFFSET + time_bin

        # Break if less ticks than the biggest bin remain, this is to avoid excessive event creation
        if ticks < config.TIME_BINS[-1]:
            break
