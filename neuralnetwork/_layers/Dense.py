import numpy as np
from neuralnetwork.utils import relu , sigmoid , softmax
from neuralnetwork.utils import relu_backward , sigmoid_backward , soft_backward
class Dense:

    """
    Implementing the input dense (fully connected) layer of a CNN.
    """

    def __init__(self, num_neurons, previous_layer, activation_function="relu"):

        """
        num_neurons: Number of neurons in the dense layer.
        previous_layer: Reference to the previous layer.
        activation_function: Name of the activation function to be used in the current layer.
        """

        if num_neurons <= 0:
            raise ValueError("Number of neurons cannot be <= 0. Please pass a valid value to the 'num_neurons' parameter.")

        # Number of neurons in the dense layer.
        self.num_neurons = num_neurons

        # Validating the activation function
        if (activation_function == "relu"):
            self.activation = relu
        elif (activation_function == "sigmoid"):
            self.activation = sigmoid
        elif (activation_function == "softmax"):
            self.activation = softmax
        else:
            raise ValueError("The specified activation function '{activation_function}' is not among the supported activation functions {supported_activation_functions}. Please use one of the supported functions.".format(activation_function=activation_function, supported_activation_functions=supported_activation_functions))

        self.activation_function = activation_function

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer

        # Initializing the weights of the layer.
        self.initial_weights = np.random.randn(self.num_neurons , self.previous_layer.layer_output_size[0]
                                                  ) * np.sqrt(2 / (self.previous_layer.layer_output_size[0] + self.previous_layer.layer_output_size[0] ))
        self.initial_bias  = np.zeros((self.num_neurons , 1))
        # The trained weights of the layer. Only assigned a value after the network is trained (i.e. the train_network() function completes).
        # Just initialized to be equal to the initial weights
        self.trained_weights = self.initial_weights.copy()
        self.trained_bias    = self.initial_bias.copy()
        # Size of the input to the layer.
        self.layer_input_size = self.previous_layer.layer_output_size
        # Size of the output from the layer.
        self.layer_output_size = ( num_neurons , self.previous_layer.layer_output_size[1] )
        # The layer_output attribute holds the latest output from the layer.
        self.layer_output = None

    def dense_layer(self, layer_input):

        """
        Calculates the output of the dense layer.

        layer_input: The input to the dense layer
        The dense_layer() method saves its result in the layer_output attribute.
        """
        self.layer_input = layer_input
        if self.trained_weights is None:
            raise TypeError("The weights of the dense layer cannot be of Type 'None'.")


        sop = numpy.matmul(self.trained_weights , self.layer_input)
        self.Z   = sop + self.trained_bias


        self.layer_output = self.activation(self.Z)
    def dense_layer_back(self , dA , Last = False):
        """
        back prop of dense layer
        dA : the gradient of the next laye
        """
        # dZ = dA * gdash(Z)
        if Last == True :
            dZ = dA
        elif self.activation is relu:
            dZ = relu_backward(dA  , self.layer_output)
        elif self.activation == sigmoid:
            dZ = sigmoid_backward(dA , self.layer_output)
        else:
            dZ = soft_backward(dA , self.layer_output)
        # The number of samples
        m  = self.layer_input_size[1]
        # Calculate the gradints
        self.dW = 1 / m * numpy.dot(dZ , self.layer_input.T)
        self.db = 1 / m * numpy.sum(dZ , keepdims = True , axis = 1)
        dA = numpy.dot(self.trained_weights.T , dZ )
        return dA
