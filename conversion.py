import mido
import os
import numpy as np

import config


"""
Returns an event sequence representation of the MIDI file associated with the passed path
"""
def load_midi(filename):

    cached_midi = os.path.join(config.CACHE_DIR, filename + '.npy')

    try:
        # Try to load cache from previous version
        events = np.load(cached_midi)
    
    except:
        # Load using mido and convert to representation
        midi = mido.MidiFile(filename)
        events = rep_from_midi(midi.ticks_per_beat, mido.merge_tracks(midi.tracks))

        # Cache the event sequence to avoid having to wait next time
        os.makedirs(os.path.dirname(cached_midi), exist_ok=True)
        np.save(cached_midi, events)
    
    return events


"""
Convert from MIDI to the representation defined in the report
"""
def rep_from_midi(tpb, midi):

    events = []
    tempo = 120
    prev_velocity = None
    
    for event in midi:
        
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


"""
Saves the passed event sequence representation as a MIDI file with name filename
"""
def save_midi(filename, events):

    os.makedirs(config.COMPOSITIONS_DIR, exist_ok=True)
    midi = midi_from_rep(events)
    midi.save(config.COMPOSITIONS_DIR + '/' + filename + '.mid')


"""
Builds a track from an iterator of passed events
"""
def midi_from_rep(events):
        
    prev_velocity = 0
    time = 0
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    held_notes = set()        
    
    for event in events:

        """
        Use offsets to determine MIDI action to take for this event
        """
        if event >= config.VELOCITY_OFFSET:
            
            prev_velocity = (event - config.VELOCITY_OFFSET) * (config.VELOCITY_RANGE // config.VELOCITY_BINS)
        
        elif event >= config.TIME_OFFSET:

            time_bin = event - config.TIME_OFFSET
            seconds = config.TIME_BINS[time_bin] / config.TICKS_PER_SEC
            time += int(mido.second2tick(seconds, midi.ticks_per_beat, mido.bpm2tempo(120)))
        
        else:

            if prev_velocity == 0:

                # Release a note if it is held
                if event in held_notes:

                    track.append(mido.Message('note_off', note=event, time=time))
                    held_notes.remove(event)
                    time = 0

            else:

                track.append(mido.Message('note_on', note=event, time=time, velocity=prev_velocity))
                held_notes.add(event)
                time = 0

    return midi.tracks.append(track)