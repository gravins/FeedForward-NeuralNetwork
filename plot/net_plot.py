import matplotlib
import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot
from matplotlib.colors import Normalize
from math import cos, sin, atan
from preprocessing.enums import NeuronType


class PlotNeuron():
    def __init__(self, x, y, neuron_weights, neuron_state, neuron_type):
        self.x_pos = x
        self.y_pos = y
        self.neuront_type = 0
        self.neuron_weights = neuron_weights
        self.neuron_state = neuron_state
        self.neuron_type = neuron_type
        self.neuron_color = self.__initialise_color(neuron_type)
        self.neuron_text_color = self.__initialise_text_color(neuron_type)

    def __initialise_color(self, neuron_type):
        # funny python implementation of switch
        switcher = {
            NeuronType.input_neuron: '#6002ee',
            NeuronType.inner_neuron: '#90ee02',
            NeuronType.output_neuron: '#C62828',
            NeuronType.bias_neuron: '#FFEA00'
        }
        return switcher.get(neuron_type, 'blue')

    def __initialise_text_color(self, neuron_type):
        # funny python implementation of switch
        switcher = {
            NeuronType.input_neuron: 'white',
            NeuronType.inner_neuron: 'black',
            NeuronType.output_neuron: 'white',
            NeuronType.bias_neuron: 'black'
        }
        return switcher.get(neuron_type, 'blue')

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x_pos, self.y_pos), radius=neuron_radius,
                               edgecolor='black', facecolor=self.neuron_color,
                               alpha=0.8, zorder=10)

        pyplot.text(
            self.x_pos, self.y_pos, "{:.2f}".format(self.neuron_state), zorder=10,
            horizontalalignment='center', verticalalignment='center',
            color=self.neuron_text_color, fontsize=8)

        pyplot.gca().add_patch(circle)

        # pyplot.gca().annotate(self.neuron_state, xy=(self.x_pos, self.y_pos), fontsize=8, ha="center")


class PlotLayer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, color,
                 layer_weights, layer_states, bias_weights, layer_type):
        self.vertical_distance_between_layers = 7
        self.horizontal_distance_between_neurons = 4
        self.neuron_radius = 1.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.color = color
        self.layer_type = layer_type
        self.layer_weights = layer_weights
        self.layer_states = layer_states
        self.bias_weights = bias_weights
        self.previous_layer = self.__get_previous_layer(network)
        self.y_pos = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x_pos = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for neur_element in range(number_of_neurons):
            neuron = PlotNeuron(x_pos, self.y_pos, self.layer_weights[neur_element],
                                self.layer_states[neur_element], self.layer_type)
            neurons.append(neuron)
            x_pos += self.horizontal_distance_between_neurons
        self.bias_neuron = PlotNeuron(x_pos, self.y_pos, self.bias_weights, 1, NeuronType.bias_neuron)

        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return (
            self.horizontal_distance_between_neurons *
            (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2)

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y_pos + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if network.layers:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, neuron1_position):
        angle = atan((neuron2.x_pos - neuron1.x_pos) / float(neuron2.y_pos - neuron1.y_pos))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)

        alpha = 1 - (sum(self.color.to_rgba(neuron2.neuron_weights[neuron1_position])) - 1) / 3
        alpha = min(1, alpha + 0.2)

        line = pyplot.Line2D(
            (neuron1.x_pos - x_adjustment, neuron2.x_pos + x_adjustment),
            (neuron1.y_pos - y_adjustment, neuron2.y_pos + y_adjustment),
            color=self.color.to_rgba(neuron2.neuron_weights[neuron1_position], alpha=alpha))
        pyplot.gca().add_line(line)

    def draw(self, layer_type=0):
        for num, neuron in enumerate(self.neurons):
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, num)
                self.__line_between_two_neurons(neuron, self.previous_layer.bias_neuron, num)
            neuron.draw(self.neuron_radius)
        if layer_type != -1:
            self.bias_neuron.draw(self.neuron_radius)

        # write Text
        x_text = - self.number_of_neurons_in_widest_layer * (self.horizontal_distance_between_neurons - 1)
        if layer_type == 0:
            pyplot.text(x_text, self.y_pos, 'Input Layer', fontsize=12)
        elif layer_type == -1:
            pyplot.text(x_text, self.y_pos, 'Output Layer', fontsize=12)
        else:
            pyplot.text(x_text, self.y_pos, 'Hidden Layer ' + str(layer_type), fontsize=12)


class PlotNeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer, max_weight, min_weight):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layer_type = 0
        self.max_weight = max_weight
        self.min_weight = min_weight

        weight_max = max(abs(max_weight), abs(min_weight))
        weight_max = max(1, weight_max)
        self.color = cm.ScalarMappable(norm=Normalize(vmin=-weight_max, vmax=weight_max), cmap=cm.seismic)

    def add_layer(self, number_of_neurons, weights, states, bias, layer_type=NeuronType.inner_neuron):

        if not self.layers:
            layer_type = NeuronType.input_neuron

        layer = PlotLayer(
            self,
            number_of_neurons,
            self.number_of_neurons_in_widest_layer,
            self.color,
            weights,
            states,
            bias,
            layer_type)

        self.layers.append(layer)

    def draw(self, params, save_name):

        fig = pyplot.figure()
        fig.patch.set_facecolor("gray")
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                i = -1
            layer.draw(i)

        x_text = self.number_of_neurons_in_widest_layer * 5
        y_text = 0
        for i in list(params.items()):
            pyplot.text(x_text, y_text, i[0] + ": " + str(i[1]), fontsize=6)
            y_text = y_text + 1

        pyplot.text(x_text, y_text + 1, "Max weight: " + str(self.max_weight), fontsize=6)
        pyplot.text(x_text, y_text + 2, "Min weight: " + str(self.min_weight), fontsize=6)

        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title('Neural Network architecture', fontsize=15)
        # pyplot.show(block=False)

        pyplot.savefig(save_name + ".png", dpi=400)


class DrawNN():

    @staticmethod
    def draw(neural_network, params, save_name):

        dims = params["inner_dimension"].copy()
        dims.insert(0, neural_network.input_dim)

        widest_layer = max(dims) + 1

        def flatten(l):
            return [it for sublist in l for item in sublist for it in item]
        flat_net = flatten(neural_network.weights)
        max_weight = max(flat_net)
        min_weight = min(flat_net)

        network = PlotNeuralNetwork(widest_layer, max_weight, min_weight)
        for pos, val in enumerate(dims[0:-1]):
            network.add_layer(val, neural_network.weights[pos], neural_network.net[pos], neural_network.bias[pos])

        network.add_layer(
            dims[-1], np.zeros(dims[-1]), neural_network.states[-1],
            np.zeros(dims[-1]), layer_type=NeuronType.output_neuron)

        network.draw(params, save_name)
