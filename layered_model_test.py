from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from layered_model import define_dense_model_single_layer, define_dense_model_with_hidden_layer
from layered_model import fit_mnist_model_single_digit, evaluate_mnist_model_single_digit
from layered_model import binarize_labels, get_mnist_data

def test_define_dense_model_single_layer():
    model = define_dense_model_single_layer(43, activation_f='sigmoid', output_length=1)    # create a model with 43 inputs, sigmoid activation function, and 1 output.
    assert len(model.layers) == 1, " model should have 1 layer"     # check if the model has 1 layer.
    assert model.layers[0].input_shape == (None, 43), " input_shape is not correct"     # check if the input_shape is correct.
    assert model.layers[0].output_shape == (None, 1), " output_shape is not correct"    # check if the output_shape is correct.


def test_define_dense_model_with_hidden_layer():
    # create a model with 43 inputs, activation function: relu - hidden layer, sigmoid - output layer, 11 neurons in the hidden layer, and 1 output.
    model = define_dense_model_with_hidden_layer(43, activation_func_array=['relu','sigmoid'], hidden_layer_size=11, output_length=1)
    assert len(model.layers) == 2, " model should have 2 layers"    # check if the model has 2 layers.
    assert model.layers[0].input_shape == (None, 43), " input_shape is not correct"
    assert model.layers[0].output_shape == (None, 11), " output_shape is not correct"
    assert model.layers[1].output_shape == (None, 1), " output_shape is not correct"


def test_fit_and_predict_mnist_single_digit_one_neuron():
    model = define_dense_model_single_layer(28*28, activation_f='sigmoid', output_length=1)     # create a model with 28*28 inputs, sigmoid activation function, and 1 output.
    (x_train, y_train), (x_test, y_test) = get_mnist_data() # get the mnist data.
    model = fit_mnist_model_single_digit(x_train, y_train, 2, model, epochs=5, batch_size=128)  # fit the model to the data.
    loss, accuracy = evaluate_mnist_model_single_digit(x_test, y_test, 2, model)    # evaluate the model on the test data.
    assert accuracy > 0.9, " accuracy should be greater than 0.9"
    loss, accuracy = evaluate_mnist_model_single_digit(x_test, y_test, 3, model)    # evaluate the model on the test data for a different digit.
    assert accuracy < 0.9, " accuracy should be smaller than 0.9"   # check if the accuracy is less than 0.9.
