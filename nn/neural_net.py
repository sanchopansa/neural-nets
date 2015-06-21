import itertools as it
import numpy as np


class NeuralNetwork(object):
    """'Abstract' implementation of a neural network. Concrete implementations
    must provide the logic for the compute_output and train methods
    """
    def __init__(self, num_inputs, num_outputs, hidden_layers,
                 activation_funcs, weight_min=-0.1, weight_max=0.1):
        """Initialize the network

        :param int num_inputs: number of inputs to the network
        :param int num_outputs: number of outputs of the network
        :param tuple hidden_layers: number of neurons in each hidden layer of
            the network. The first element in the tuple represents the number
            of neurons in the layer immediately above the input layer
        :param tuple activation_funcs: activation functions for each layer of
        the network. If a single element is passed, it is assumed that the same
        activation function should be used for all layers
        :param float weight_min: optional mininum weight to use when
            initializing the weight matrix with random values
        :param float weight_max: optional maximum weight to use when
            initializing the weight matrix with random values
        :returns: initialized neural network instance
        """
        if len(activation_funcs) != 1 and \
           len(hidden_layers) + 1 != len(activation_funcs):
            raise ValueError("Invalid number of activation functions. "
                             "Activation functions should be available "
                             "for all layers except the input layer.")

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = hidden_layers
        self._activation_funcs = activation_funcs
        self.weight_min = weight_min
        self.weight_max = weight_max
        self._num_weights = None
        self._weights = None

    @property
    def weights(self):
        if self._weights is None:
            self._weights = self.random_weights(self.weight_min,
                                                self.weight_max)
        return self._weights

    @weights.setter
    def weights(self, weights):
        assert len(weights) == self.num_weights, \
            "Wrong number of weights. Missing weight for bias neurons?"
        self._weights = np.array(weights)

    @property
    def activation_funcs(self):
        if len(self.hidden_layers) + 1 == len(self._activation_funcs):
            # Activation functions provided separately for each layer
            return self._activation_funcs
        elif len(self._activation_funcs) == 1:
            # A single activation function provided for all layers
            return it.repeat(self._activation_funcs[0],
                             len(self.hidden_layers) + 1)
        else:
            # Should never be reached because of the check inside __init__,
            # but better be safe than sorry
            raise ValueError("Invalid number of activation functions")

    @property
    def num_weights(self):
        """Return the total number of weights for the feedforward network. The
        network is assumed to be fully connected for the sake of generality.
        All layers of the network except the output layer are assumed to have a
        bias neuron, which is also fully connected to all neurons in the next
        layer
        """
        if self._num_weights is None:
            if len(self.hidden_layers) == 0:
                # +1 for bias node
                self._num_weights = (self.num_inputs + 1) * self.num_outputs
            else:
                self._num_weights = reduce(
                    # +1 for bias node
                    lambda acc, pair: acc + (pair[0] + 1) * pair[1],
                    self.num_neurons_in_layer_pairs, 0)
        return self._num_weights

    @property
    def weight_matrices(self):
        """Generator of weight matrices, with each weight matrix representing
        the weights between a pair of adjacent layers, starting with the
        connections between the input layer and the layer above
        """
        if len(self.hidden_layers) == 0:
            yield self.weights.reshape(self.num_inputs + 1, self.num_outputs)
        else:
            total_conns = 0
            for num_neurons_a, num_neurons_b in \
                    self.num_neurons_in_layer_pairs:
                conns_bn_cur_layers = (num_neurons_a + 1) * num_neurons_b
                yield self.weights[total_conns:
                                   total_conns + conns_bn_cur_layers].reshape(
                    num_neurons_a + 1, num_neurons_b)
                total_conns += conns_bn_cur_layers

    @property
    def num_neurons_in_layer_pairs(self):
        """Return a lazy iterator of tuples, with each tuple representing the
        number of elements in pairs of adjacent layers, excluding the bias
        neurons. The first element of the sequence represents the number of
        neurons in the start layer and the layer above
        """
        return it.izip(it.chain([self.num_inputs],
                                self.hidden_layers),
                       it.chain(self.hidden_layers,
                                [self.num_outputs]))

    def random_weights(self, low=-0.1, high=0.1):
        """Return an np.array of random weights. Useful for initialization"""
        return np.random.uniform(low, high, self.num_weights)

    def compute_output(self):
        # Implementation must be provided in subclasses
        raise NotImplementedError

    def train(self):
        # Implementation must be provided in subclasses
        raise NotImplementedError
