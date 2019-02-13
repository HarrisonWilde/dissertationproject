import numpy
# https://github.com/vishnubob/python-midi/
from midi import *

lowerBound = 24
upperBound = 102

# Can deal with multi-channel midi, although my network focusses on single channel midi
def fromMIDI(midifile, verbose=False):

    pattern = read_midifile(midifile)
    timeleft = [track[0].tick for track in pattern]
    posns = [0 for track in pattern]

    statematrix = []
    span = upperBound - lowerBound
    time = 0

    state = [[0,0] for x in range(span)]
    statematrix.append(state)

    while pattern.resolution < 4:
        if verbose:
            print "Doubling resolution..."
        pattern.resolution = pattern.resolution * 2
        for i in range(0, len(pattern[0])):
            if isinstance(pattern[0][i], NoteEvent):
                pattern[0][i].tick = pattern[0][i].tick * 2
            elif isinstance(pattern[0][i], SetTempoEvent):
                pattern[0][i].set_bpm(pattern[0][i].get_bpm() / 2)

    while True:
        # Check to see if note grain has ended, depending on resolution of the pattern; new created state defaults to holding previous notes
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            statematrix.append(state)

        for i in range(len(timeleft)):

            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]
                evt = track[pos]
                if isinstance(evt, NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                    else:
                        if isinstance(evt, NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch - lowerBound] = [0, 0]
                        else:
                            state[evt.pitch - lowerBound] = [1, 1]

                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    return statematrix

def toMIDI(statematrix, name="untitled"):

    statematrix = numpy.asarray(statematrix)
    pattern = Pattern()
    track = Track()
    pattern.append(track)

    span = upperBound - lowerBound
    tickscale = 55

    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        rels = []
        atks = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    rels.append(i)
                elif n[1] == 1:
                    rels.append(i)
                    atks.append(i)
            elif n[0] == 1:
                atks.append(i)
        for pitch in rels:
            track.append(NoteOffEvent(tick=(time - lastcmdtime)*tickscale, pitch=pitch+lowerBound))
            lastcmdtime = time
        for pitch in atks:
            track.append(NoteOnEvent(tick=(time - lastcmdtime)*tickscale, velocity=40, pitch=pitch+lowerBound))
            lastcmdtime = time
        
        prevstate = state

    track.append(EndOfTrackEvent(tick=1))

    write_midifile("{}.mid".format(name), pattern)