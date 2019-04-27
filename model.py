import numpy as np
import torch
from torch.nn import Module, Linear, RNN, LSTM, GRU
from torch.nn.functional import softmax

import config


"""
Relativistic dilated LSTM or GRU network architecture as defined in the final report, can be given different unit and layer arguments on creation
"""
class Model(Module):

    def __init__(self, layers, recurrent_unit, units, mood_units):
        
        super(Model, self).__init__()

        self.units = units
        self.layers = layers
        self.mood_units = mood_units
        self.dilations = [2 ** i for i in range(self.layers)]

        self.mood_layer = Linear(config.MOOD_DIMENSION, self.mood_units)

        if recurrent_unit == 'LSTM':
            RU = LSTM
        elif recurrent_unit == 'GRU':
            RU = GRU
        else:
            raise NotImplementedError

        first_hidden_layer = [RU(config.FULL_RANGE + self.mood_units, self.units, batch_first=True)]
        dilated_hidden_layers = [RU(self.units, self.units, batch_first=True) for i in range(1, self.layers)]
        self.hidden_layers = first_hidden_layer + dilated_hidden_layers
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            self.add_module('D' + recurrent_unit + str(i), hidden_layer)     

        self.output_layer = Linear(self.units, config.FULL_RANGE)

    """
    Forward step for the network inputting passed tensors of inputs and moods
    """
    def forward(self, inputs, moods, states=None):
        
        batch_size, sequence_length, _ = inputs.size()

        # Distributed mood representation
        moods = self.mood_layer(moods)
        moods = moods.unsqueeze(1).expand(batch_size, sequence_length, self.mood_units)
        inputs = torch.cat((inputs, moods), dim=2)

        if states is None:
            states = [None] * self.layers

        for i, hidden_layer in enumerate(self.hidden_layers):
            
            prev_inputs = inputs
            dilation = self.dilations[i]

            # Usually during generation
            if sequence_length == 1:

                if states[i] is None:
                    states[i] = (0, tuple(None for _ in range(dilation)))

                step, dilated_states = states[i]
                inputs, dilated_state = hidden_layer(inputs, dilated_states[step % dilation])
                states[i] = (step + 1, dilated_states[:step % dilation] + (dilated_state,) + dilated_states[step % dilation + 1:])

            else:
                # Reshape to spread across dilated skip connections
                inputs = (inputs
                    .unfold(1, dilation, dilation)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .view(batch_size * dilation, sequence_length // dilation, -1))
                
                inputs, states[i] = hidden_layer(inputs, states[i])

                # inputs are now of dimension [batch_size * dilation, sequence_length // dilation, features], need to return it to its original shape
                inputs = (inputs
                    .contiguous()
                    .view(batch_size, dilation, sequence_length // dilation, -1)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .view(batch_size, sequence_length, -1))

            if i > 0:
                inputs = prev_inputs + inputs

        outputs = self.output_layer(inputs)
        
        return outputs, states

    """
    Returns a vector of probabilities to be used in generating new compositions based on previous states
    """
    def compose(self, inputs, moods, states, temperature):

        inputs, states = self.forward(inputs, moods, states)
        length = inputs.size(1)
        inputs = softmax(inputs.view(-1, config.FULL_RANGE) / temperature, dim=1)
        inputs = inputs.view(-1, length, config.FULL_RANGE)

        return inputs, states
