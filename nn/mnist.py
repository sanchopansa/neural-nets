import itertools as it

from nn.neural_net import SimpleNN
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
import numpy as np


def stop_training(num_epochs, error):
    print "num_epochs: %d" % num_epochs
    print "error: %f" % error
    return num_epochs == 25 or error == 0


def compute_error(actual, expected):
    return 1 if expected != actual else 0


def select_output(output_values):
    return output_values.index(max(output_values))


if __name__ == '__main__':
    # Stored by default in ~/scikit_learn_data
    mnist = fetch_mldata('MNIST original')
    images = mnist.data
    labels = mnist.target

    x_train, x_test, y_train, y_test = train_test_split(images, labels,
                                                        test_size=0.4,
                                                        random_state=0)
    print "Number of training examples: %d" % len(x_train)
    nn = SimpleNN(28 * 28, 10, [], [lambda x: x], select_output, -0.6, 0.6)
    nn.train(x_train, y_train, 0.15, compute_error, stop_training)

    print "Number of test examples: %d" % len(x_test)
    error = 0
    for x, y in it.izip(x_test, y_test):
        y_est = nn.compute_output(np.append(x, 1))
        error += compute_error(y_est, y)
    print "Number of misclassified examples: %d" % error

    if __debug__:
        matrix = next(nn.weight_matrices)
        zero = matrix[0, :-1].reshape(28, 28)
        one = matrix[1, :-1].reshape(28, 28)
        plt.figure()
        plt.imshow(zero)
        plt.figure()
        plt.imshow(one)
        plt.show()
