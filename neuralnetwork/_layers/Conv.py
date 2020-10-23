import numpy as np
from neuralnetwork.utils import single_conv
from neuralnetwork.utils import relu , sigmoid
from neuralnetwork.utils import relu_backward , sigmoid_backward

class Conv2D:

    """
    Implementing the convolution layer.
    """

    def __init__(self, num_filters, kernel_size, previous_layer, activation_function=None):

        """
        num_filters: Number of filters in the convolution layer.
        kernel_size: Kernel size of the filter.
        previous_layer: A reference to the previous layer.
        activation_function=None: The name of the activation function to be used in the conv layer. If None, then no activation function is applied besides the convolution operation. The activation function can be applied by a separate layer.
        """

        if num_filters <= 0:
            raise ValueError("Number of filters cannot be <= 0. Please pass a valid value to the 'num_filters' parameter.")
        # Number of filters in the conv layer.
        self.num_filters = num_filters

        if kernel_size <= 0:
            raise ValueError("The kernel size cannot be <= 0. Please pass a valid value to the 'kernel_size' parameter.")
        # Kernel size of each filter.
        self.kernel_size = kernel_size

        # Validating the activation function
        if (activation_function is None):
            self.activation = None
        elif (activation_function == "relu"):
            self.activation = relu
        elif (activation_function == "sigmoid"):
            self.activation = sigmoid
        elif (activation_function == "softmax"):
            raise ValueError("The softmax activation function cannot be used in a conv layer.")
        else:
            raise ValueError("The specified activation function '{activation_function}' is not among the supported activation functions {supported_activation_functions}. Please use one of the supported functions.".format(activation_function=activation_function, supported_activation_functions=supported_activation_functions))

        # The activation function used in the current layer.
        self.activation_function = activation_function

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer

        # A reference to the bank of filters.
        self.filter_bank_size = (self.kernel_size,
                                 self.kernel_size,
                                 self.previous_layer.layer_output_size[-1] ,
                                 self.num_filters)
        # initialize bais

        # Initializing the filters of the conv layer.
        self.initial_weights = np.random.randn(self.filter_bank_size[0] ,
                                                 self.filter_bank_size[1] ,
                                                 self.filter_bank_size[2] ,
                                                 self.filter_bank_size[3]) * 0.01

        # The trained filters of the conv layer. Only assigned a value after the network is trained (i.e. the train_network() function completes).
        # Just initialized to be equal to the initial filters
        self.trained_weights = self.initial_weights.copy()

        # Size of the input to the layer.

        self.initial_bias = np.zeros((1 , 1 , 1 , self.num_filters ))

        self.trained_bias = self.initial_bias.copy()

        self.layer_input_size = self.previous_layer.layer_output_size

        # Size of the output from the layer.
        # Later, it must conider strides and paddings
        self.layer_output_size = (self.previous_layer.layer_output_size[0] , self.previous_layer.layer_output_size[1] - self.kernel_size + 1,
                                  self.previous_layer.layer_output_size[2] - self.kernel_size + 1,
                                  num_filters)

        # The layer_output attribute holds the latest output from the layer.
        self.layer_output = None

    def conv_(self, input2D ):

        """
        Convolves the input (input2D) by a single filter (conv_filter).

        input2D: The input to be convolved by a single filter.
        conv_filter: The filter convolving the input.

        Returns the result of convolution.
        """
        m , n_HPrev , n_WPrev , n_CPrev = input2D.shape
        stride = 1
        N_C = self.filter_bank_size[3]
        f   = self.filter_bank_size[1]
        n_W = int((n_WPrev - f) / stride + 1 )
        n_H = int((n_HPrev - f) / stride + 1)

        result = np.zeros(( m , n_H , n_W , N_C))
        # Looping through the image to apply the convolution operation.
        for i in range(m):
            Aprev = input2D[i , : , : , :]
            # Loop through the coulmns
            for h in range(n_H):
                vert_start  = h * stride
                vert_end    = h * stride + f
                # Loop through the rows of the image
                for w in range (n_W):
                    horiz_start = w * stride
                    horiz_end   = w * stride + f
                    # Loop through the num of filters
                    for c in range(N_C):
                        A_slice = Aprev[vert_start : vert_end , horiz_start : horiz_end , :]


                        result[i , h , w , c] = single_conv( A_slice ,
                                                            self.trained_weights[: , : , : , c] ,
                                                            self.trained_bias[: , : , : , c])

        self.Z = result
        if self.activation is not None: # Saving the SOP in the convolution layer feature map.
            result = self.activation(result) # Saving the activation function result in the convolution layer feature map.

        # Clipping the outliers of the result matrix.
        return result

    def conv_back(self , dA):
        """
        function to get the back prop of the conv layer => calc the gradint dW , db , dA
        dA: the gradinet of the last layer
        return dA
        """
        if self.activation == relu:
            dZ = relu_backward(dA , self.Z)
        elif self.activation == sigmoid:
            dZ = sigmoid_backward(dA , self.Z)
        elif self.activation == None:
            dZ = dA

        # dimensions of the current layer
        m , n_H , n_W , n_C     = dZ.shape
        # dimensions of the current weights
        f  , f , n_CPrev , n_C  = self.initial_weights.shape
        # dimensions of the previous layer
        m , n_HPrev , n_WPrev , n_CPrev = self.layer_input_size
        # set A to the input of the layer
        A = self.input2D
        # weights of the layer
        W = self.trained_weights
         # initial gradiants of W , b and A previous
        self.stride = 1
        self.dW = np.zeros((f , f , n_CPrev , n_C ))
        self.db = np.zeros((1 , 1 , 1 ,n_C))
        dAprev  = np.zeros((m , n_HPrev , n_WPrev , n_CPrev))

        for i in range(m):
            A  = self.input2D[i]
            da = dAprev[i]

            for h in range(n_H):

                for w in range(n_W):

                    for c in range(n_C):
                        vert_start = h * self.stride
                        vert_end = vert_start + f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + f

                        a_slice = A[vert_start:vert_end, horiz_start:horiz_end, :]

                        da[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]

                        self.dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        self.db[:,:,:,c] += dZ[i , h , w , c]
            dAprev[i , : , : , :] = da[: , : , : ]

        return dAprev

    def conv(self, input2D):

        """
        Convolves the input (input2D) by a filter bank.

        input2D: The input to be convolved by the filter bank.
        The conv() method saves the result of convolving the input by the filter bank in the layer_output attribute.
        """
        if len(input2D.shape) != 4:
            raise ValueError("Number of dimensions is not vaild , expected 4 but {Shape} found".format(Shape = input2D.shape))
        if self.initial_weights.shape[1]%2==0: # Check if filter diemnsions are odd.
            raise ValueError('A filter must have an odd size. I.e. number of rows and columns must be odd.')
        self.input2D = input2D
        self.layer_output = self.conv_(self.input2D)
