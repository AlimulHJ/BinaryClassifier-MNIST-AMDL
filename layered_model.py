from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def define_dense_model_single_layer(input_length, activation_f='sigmoid', output_length=1):
    """Define a dense model with a single layer with the following parameters:
    input_length: the number of inputs
    activation_f: the activation function
    output_length: the number of outputs (number of neurons)"""

    model = keras.Sequential()  # create a sequential model.
    model.add(layers.Input(shape=(input_length,)))  # add an input layer.
    model.add(layers.Dense(output_length, activation=activation_f))     # add a dense layer(most common & frequently used).
    # Dense layer is a regular densely-connected NN layer. https://keras.io/api/layers/core_layers/dense/

    return model

def define_dense_model_with_hidden_layer(input_length, 
                                         activation_func_array=['relu','sigmoid'],
                                         hidden_layer_size=10,
                                         output_length=1):
    """Define a dense model with a hidden layer with the following parameters:
    input_length: the number of inputs
    activation_func_array: the activation function for the hidden layer and the output layer
    hidden_layer_size: the number of neurons in the hidden layer
    output_length: the number of outputs (number of neurons in the output layer)"""

    model = keras.Sequential()  # create a sequential model - a linear stack of layers. https://keras.io/guides/sequential_model/
    model.add(layers.Input(shape=(input_length,)))  # add an input layer.
    model.add(layers.Dense(hidden_layer_size, activation=activation_func_array[0]))  # add a hidden layer.
    model.add(layers.Dense(output_length, activation=activation_func_array[1]))  # add an output layer.

    return model


def get_mnist_data():
    """Get the MNIST data."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()    # load the mnist data.
    x_train = x_train.reshape(60000, 784).astype('float32') / 255   # reshape the train data.
    x_test = x_test.reshape(10000, 784).astype('float32') / 255     # reshape the test data.
    return (x_train, y_train), (x_test, y_test)

def binarize_labels(labels, target_digit=2):
    """Binarize the labels."""
    labels = 1*(labels==target_digit)   # binarize the labels by checking if the label is equal to the target_digit.
    # How this Binarizatio works:
    # if the label is equal to the target_digit, it will return 1, else it will return 0.
    # e.g: if target_digit 2, then labels will be binarized as follows: 2 -> 1, 3 -> 0, 4 -> 0, 5 -> 0,...

    return labels

def fit_mnist_model_single_digit(x_train, y_train, target_digit, model, epochs=10, batch_size=128):
    """Fit the model to the data.
    compile the model and add parameters for  the "optimizer", the loss function , 
    and the metrics, Hint: use binary crossentropy for the loss function .

    then fit the model on the training data. (pass the epochs and batch_size params)
    """
    y_train = binarize_labels(y_train, target_digit)    # binarize the labels for the target_digit.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # compile the model.
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)   # fit the model on the training data.

    return model

def evaluate_mnist_model_single_digit(x_test, y_test, target_digit, model):
    """Evaluate the model on the test data.
    Hint: use model.evaluate() to evaluate the model on the test data.
    """
    y_test = binarize_labels(y_test, target_digit)  # binarize the labels for the target_digit.
    loss, accuracy = model.evaluate(x_test, y_test) # evaluate the model on the test data.

    return loss, accuracy