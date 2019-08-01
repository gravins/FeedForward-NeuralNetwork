from enum import Enum


class NeuronType(Enum):
    input_neuron = 0
    output_neuron = 1
    inner_neuron = 2
    bias_neuron = 3

class TaskType(Enum):
    regression = 0
    classification = 1
