import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


def window_transform_series(series, window_size):

    """This function transforms a given series into input/output pairs.
    Args:
        series: 'Numpy array' of shape (size,1)
        window_size: 'int' specifying the size of each input pair
        (Stride or step is assumed as one)
    Returns:
        X,y : Pairs of input/output as 'Numpy Arrays'
    """
    # containers for input/output pairs
    X = []
    y = []

    ## Total number of inputs possible is given by len(series) - window_size
    X = np.asarray(X)
    X = np.zeros(((len(series) - window_size),window_size))

    ## Assign the series values to the corresponding X's
    for i in range(len(X)):
        X[i][0:window_size] = series[i:i+window_size]

    ## The next value in series is assigned as output (or y)
    y = series[window_size:]
    y = np.asarray(y)
    y = np.reshape(y, (len(y),1)) #optional

    assert(type(X).__name__ == 'ndarray')
    assert(type(y).__name__ == 'ndarray')
    assert(len(X)==len(y))

    return X,y

def build_part1_RNN(window_size):
    """ Builds an RNN for regression on given time series input/output data
    Args:
        window_size: 'int' as input shape for given time series

    Returns:
        model: A Sequential 'model in Keras' which has one LSTM layer and one Dense layer without any activation function (as the activation is linear)
    """

    ## 5 hidden units of LSTM connected to a fully connected layer with 1 unit
    model = Sequential([
        LSTM(5, input_shape=(window_size,1)),
        Dense(1),
    ])
    return model


def cleaned_text(text):
    """ Returns the text input with only ascii lowercase and the punctuation given below included.
    Args:
        text: 'List' of chars with punctuations (all chars already in lowercase)

    Returns:
        text: 'List' of chars with punctuations replaced by spaces
    """
    punctuation = ['!', ',', '.', ':', ';', '?']
    # unique_chars = sorted(list(set(text)))

    all_text = []
    from string import ascii_lowercase
    for char in text:
        if (char in ascii_lowercase) or (char in punctuation):
            all_text.append(char)
        else:
            all_text.append(" ")

    # text_all = ''.join(all_text)
    return ''.join(all_text)


def window_transform_text(text, window_size, step_size):
    """Transforms the input text and window-size into a set of input/output pairs for use with our RNN model
    Args:
        text: 'List' of chars
        window_size: 'int' specifying the size of each input block
        step_size: 'int' specifying the step or stide with which the input pair is to be generated

    Returns:
        inputs: 'List' of chars
        output: 'List' of char which follows the input string of chars
    """

    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text), step_size):
        try:
            ## Update the outputs first because if there is no character in text then the try loop will go in except
            outputs.append(text[i+window_size])
            inputs.append(text[i:i+window_size])
        except:
            break
    return inputs,outputs

# TODO build the required RNN model:
#
def build_part2_RNN(window_size, num_chars):
    """A single LSTM hidden layer with softmax activation, categorical_crossentropy loss
    Args:
        window_size: 'int' as first dimension for input shape
        num_chars: 'int' as second dimension for input shape
    Returns:
        model: A Keras model of RNN with a single LSTM hidden layer
    """
    model = Sequential([
        LSTM(200, input_shape=(window_size,num_chars)),
        Dense(num_chars),
        Activation('softmax'),
    ])
    return model
