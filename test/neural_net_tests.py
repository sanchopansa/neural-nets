import itertools as it

from nose.tools import raises
import numpy as np

from nn import NeuralNetwork


@raises(ValueError)
def test_nn_init_invalid():
    NeuralNetwork(10, 10, [], [])


def test_nn_init_with_single_activation_function():
    NeuralNetwork(10, 2, (5, 3), range(1))


def test_weights_init():
    # Pass in a list instead of actual activation for test
    weight_min = -0.5
    weight_max = 0.5
    nn = NeuralNetwork(10, 2, [], range(1),
                       weight_min=weight_min, weight_max=weight_max)
    assert all([weight_min <= weight <= weight_max for weight in nn.weights])


def test_num_neurons_in_layer_pairs_without_hidden_layers():
    nn = NeuralNetwork(5, 2, [], range(1))
    assert list(nn.num_neurons_in_layer_pairs) == [(5, 2)]


def test_num_neurons_in_layer_pairs_with_hidden_layers():
    nn = NeuralNetwork(15, 2, (10, 5), range(3))
    assert list(nn.num_neurons_in_layer_pairs) == [(15, 10), (10, 5), (5, 2)]


def test_num_weights_without_hidden_layers():
    nn = NeuralNetwork(100, 7, [], range(1))
    assert nn.num_weights == 707


def test_num_weights_with_hidden_layers():
    nn = NeuralNetwork(10, 2, (4, 3), range(3))
    assert nn.num_weights == (44 + 15 + 8)


def test_activation_funcs():
    x_plus_one = lambda x: x + 1
    nn = NeuralNetwork(50, 2, (20, 10, 5), [x_plus_one])
    assert list(nn.activation_funcs) == [x_plus_one] * 4

    x_plus_two = lambda x: x + 2
    nn = NeuralNetwork(50, 2, (20,), [x_plus_one, x_plus_two])
    assert list(nn.activation_funcs) == [x_plus_one, x_plus_two]


def test_weight_matrix_without_hidden_layers():
    weights = np.random.uniform(size=200)
    matrix = weights.reshape(50, 4)
    nn = NeuralNetwork(49, 4, [], range(1))
    nn.weights = weights

    assert np.all(np.equal(list(nn.weight_matrices)[0], matrix))


def test_weight_matrix_with_hidden_layers():
    weights_a = np.random.uniform(0, 1, 55)
    weights_b = np.random.uniform(0, 1, 12)
    matrix_a = weights_a.reshape((11, 5))
    matrix_b = weights_b.reshape((6, 2))

    nn = NeuralNetwork(10, 2, (5, ), range(2))
    nn.weights = list(it.chain(weights_a, weights_b))

    for m1, m2 in zip(nn.weight_matrices, [matrix_a, matrix_b]):
        assert np.all(np.equal(m1, m2))
