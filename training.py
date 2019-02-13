import os, conversion, cPickle, random, numpy, time, warnings
from data import noteStateMatrixToInputForm

# PLAY WITH THIS
batch_width = 10	# number of sequences in a batch
batch_len = 16*8	# length of each sequence
division_len = 16	# interval between possible start locations


def importMIDI(path):

    MIDI = {}

    for filename in os.listdir(path):

        if filename[-4:].lower() != ".mid":
        	raise NotMidiFileError("Not a valid .mid file")

        name = filename[:-4]        
        matrix = conversion.fromMIDI(os.path.join(path, filename), verbose=True)
	    
        # Only add to MIDI dict if the resulting matrix is an appropriate size to make up at least one full training sequence
        if len(matrix) > batch_len:
	        MIDI[name] = matrix
	        print "Imported pattern named: " + name
        else:
            warnings.warn(name + " is too short for training purposes, remove it from your dataset")

    return MIDI


def getMIDI(path):

    filename = random.choice(os.listdir(path))
    name = filename[:-4]

    if filename[-4:].lower() != ".mid":
        warnings.warn(name + " is not a valid .mid file, remove it from your dataset")
        return getMIDI(path)

    matrix = conversion.fromMIDI(os.path.join(path, filename))
    
    # Only add to MIDI dict if the resulting matrix is an appropriate size to make up at least one full training sequence
    if len(matrix) > batch_len:
        return matrix
    else:
        warnings.warn(name + " is too short for training purposes, remove it from your dataset")
        return getMIDI(path)


def getSegment(inputData):

    if isinstance(inputData, str):
        choice = getMIDI(inputData)
    else:
        choice = random.choice(inputData.values())

    start = random.randrange(0, len(choice) - batch_len, division_len)
    # print "Range is {} {} {} -> {}".format(0, len(choice) - batch_len, division_len, start)

    seg_out = choice[start:start+batch_len]
    seg_in = noteStateMatrixToInputForm(seg_out)

    del choice
    return seg_in, seg_out


def getBatch(inputData):

    i,o = zip(*[getSegment(inputData) for x in range(batch_width)])
    return numpy.array(i), numpy.array(o)


def train(model, inputData, iters, name, init=0):
    
    start = time.time()

    for i in range(init, iters):
        
        error = model.update(*getBatch(inputData))

        if i % 10 == 0:
            m, s = divmod(time.time() - start, 60)
            h, m = divmod(m, 60)
            print "iteration {0}, error = {1}, time elapsed = {2:0.0f}:{3:02.0f}:{4:02.0f}".format(i, error, h, m, s)
        if i % 100 == 0:
            # output a dump of params
            cPickle.dump(model.learned_config, open("output/parameters{}{}.p".format(i, name), "wb"))

        # del error