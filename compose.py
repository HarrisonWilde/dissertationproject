import numpy as np
import torch
import torch.nn as nn
from heapq import nlargest
from operator import itemgetter
from tqdm import trange

import config
from conversion import save_midi

"""
Composes a piece using the passed inputs
"""
def compose(model, name, mood_input, temperature=0.8, n_candidates=1, sequence_length=5000):

    avg_probabilities = 1
    candidates = [(1, (), None)]
    mood = torch.FloatTensor(mood_input)
    if torch.cuda.is_available():
        mood = mood.cuda()
    mood = mood.unsqueeze(0)

    for _ in trange(sequence_length, desc=str(mood_input), leave=True):

        sum_probabilities = 0
        next_candidates = []

        # Go through each candidate and generate outputs for it to sample possible next events from
        for prev_probabilities, events, states in candidates:

            # Get next probability outputs and state from the model using previous event
            output, next_states = model.compose(extract_previous_event(events), mood, states, temperature)
            output = output.squeeze(1)

            for _ in range(n_candidates):
                
                # Sample the next chosen event from the output of the model
                sampled_event = (output.multinomial(num_samples=1).data)[0, 0]
                
                # Calculate the probability of the sampled event occurring and use it to come up with next candidates
                event_probabilities = prev_probabilities * output.data[0, sampled_event]
                next_candidates.append((event_probabilities / avg_probabilities, events + (sampled_event,), next_states))
                sum_probabilities += event_probabilities

        # Find the sequence(s) that have the highest associated probabilities
        candidates = nlargest(n_candidates, next_candidates, key=itemgetter(0))
        avg_probabilities = sum_probabilities / len(next_candidates)

    # Return the most probable sequence of events of all candidates
    chosen_events = np.array(max(candidates, key=itemgetter(0))[1])
    save_midi(name + '/' + str(mood_input), chosen_events)



"""
Creates tensors of correct dimension containing the previous event
"""
def extract_previous_event(events):

    if len(events) > 0:

        previous_event = np.zeros(config.FULL_RANGE)
        previous_event[events[-1]] = 1
        previous_event = torch.from_numpy(previous_event).float()
        if torch.cuda.is_available():
            previous_event = previous_event.cuda()
        previous_event = previous_event.unsqueeze(0)

    else:

        previous_event = torch.zeros((1, config.FULL_RANGE))
        if torch.cuda.is_available():
            previous_event = previous_event.cuda()

    return previous_event.unsqueeze(1)