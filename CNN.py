import numpy

# Supported activation functions by the cnn.py module.
supported_activation_functions = ("sigmoid", "relu", "softmax")

def sigmoid(sop):

    """
    Applies the sigmoid function.
    sop: The input to which the sigmoid function is applied.
    Returns the result of the sigmoid function.
    """

    if type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    return 1.0 / (1 + numpy.exp(-1 * sop))

def relu(sop):

    """
    Applies the rectified linear unit (ReLU) function.
    sop: The input to which the relu function is applied.
    Returns the result of the ReLU function.
    """

    if not (type(sop) in [list, tuple, numpy.ndarray]):
        if sop < 0:
            return 0
        else:
            return sop
    elif type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    result = sop
    result[sop < 0] = 0

    return result

def sigmoid_backward(dA, Z):
    
    """
    function to get dZ = dA * g(Z)
    dA : the derivative from the next layer
    Z  : Z of this layer
    """
    S = 1/(1+numpy.exp(-Z))
    dZ = dA * S * (1-S)
    return dZ

def relu_backward(dA , Z):
    
    """
    function to get dZ = dA * g(Z)
    dA : the derivative from the next layer
    Z  : Z of this layer
    """
    dZ = numpy.array(dA, copy=True) # just converting dz to a correct object.      
    dZ[Z < 0] = 0
    return dZ


def soft_backward(dA , layer_outputs ):
    
    """
    function to get dZ = dA * g(Z)
    dA : the derivative from the next layer
    Z  : Z of this layer
    """
     
    expL = numpy.exp(layer_outputs - numpy.max(layer_outputs))
    A  = expL /(expL.sum(axis = 0 , keepdims = True) + 0.0001)
    dZ =  dA * A * (1 - A)
    return dZ

def softmax(layer_outputs):

    """
    Applies the sotmax function.
    sop: The input to which the softmax function is applied.
    Returns the result of the softmax function.
    """
    expL = numpy.exp(layer_outputs - numpy.max(layer_outputs))
    return expL / (expL.sum(axis = 0 , keepdims = True) + 0.0001)

def layers_weights(model, initial=True):

    """
    Creates a list holding the weights of all layers in the CNN.
    model: A reference to the instance from the cnn.Model class.
    initial: When True, the function returns the initial weights of the layers. When False, the trained weights of the layers are returned. The initial weights are only needed before network training starts. The trained weights are needed to predict the network outputs.
    Returns a list (network_weights) holding the weights of the layers in the CNN.
    """

    network_weights = []

    layer = model.last_layer
    while "previous_layer" in layer.__init__.__code__.co_varnames:
        if type(layer) in [Conv2D, Dense]:
            # If the 'initial' parameter is True, append the initial weights. Otherwise, append the trained weights.
            if initial == True:
                network_weights.append(layer.initial_weights)
            elif initial == False:
                network_weights.append(layer.trained_weights)
            else:
                raise ValueError("Unexpected value to the 'initial' parameter: {initial}.".format(initial=initial))

        # Go to the previous layer.
        layer = layer.previous_layer

    # If the first layer in the network is not an input layer (i.e. an instance of the Input2D class), raise an error.
    if not (type(layer) is Input2D):
        raise TypeError("The first layer in the network architecture must be an input layer.")

    # Currently, the weights of the layers are in the reverse order. In other words, the weights of the first layer are at the last index of the 'network_weights' list while the weights of the last layer are at the first index.
    # Reversing the 'network_weights' list to order the layers' weights according to their location in the network architecture (i.e. the weights of the first layer appears at index 0 of the list).
    network_weights.reverse()
    return numpy.array(network_weights)

  

def update_layers_trained_weights(model, final_weights):

    """
    After the network weights are trained, the 'trained_weights' attribute of each layer is updated by the weights calculated after passing all the epochs (such weights are passed in the 'final_weights' parameter).
    By just passing a reference to the last layer in the network (i.e. output layer) in addition to the final weights, this function updates the 'trained_weights' attribute of all layers.
    model: A reference to the instance from the cnn.Model class.
    final_weights: An array of layers weights as matrices after passing through all the epochs.
    """

    layer = model.last_layer
    layer_idx = len(final_weights) - 1
    while "previous_layer" in layer.__init__.__code__.co_varnames:
        if type(layer) in [Conv2D, Dense]:
            layer.trained_weights = final_weights[layer_idx]
    
            layer_idx = layer_idx - 1

        # Go to the previous layer.
        layer = layer.previous_layer

def encoding(labels):
    """
    function to encode y_train
    labels = y_train
    """
    
    if type(labels) in [tuple , list]:
        labels = numpy.array(labels)
    
    # num of classes  
    num_unique_class = len( numpy.unique(labels) )
    # initialize the output after one hot encoding    
    solution = numpy.zeros(( max(labels) + 1 , labels.size))
    
    # loop through the target array to encode every label 
    for index in range(labels.shape[0] ):

      solution[: , index ][labels[index]] = 1

    return solution

def categorical_crossentropy(y_pred , y_target):
    """
    function to calculate the cross entropy error
    y_pred : the output of the last layer which use softmax as activation function ... shape = [10 , 250]
    target : labels
    """    
    # Calculate the cosy of this epoch
    cost = - numpy.mean([numpy.log(y_pred[: , i][y_target[i]] + 0.0000001) for i in range(len(y_target))])
    acc = sum(np.argmax(y_pred , axis = 0) == y_target) / y_target.shape[0]
    m = y_pred.shape[1]
    for index in range(m):
    	y_pred[: , index][y_target[index]] -= 1
    	    # Calculate the drivative of the last layer 
    return cost , y_pred , acc

class Input2D:

    """
    Implementing the input layer of a CNN.
    The CNN architecture must start with an input layer.
    """

    def __init__(self, input_shape):

        """
        input_shape: Shape of the input sample to the CNN.
        """

        # If the input sample has less than 2 dimensions, then an exception is raised.
        if len(input_shape) < 2:
            raise ValueError("The Input2D class creates an input layer for data inputs with at least 2 dimensions but ({num_dim}) dimensions found.".format(num_dim=len(input_shape)))
        # If the input sample has exactly 2 dimensions, the third dimension is set to 1.
        elif len(input_shape) == 2:
            input_shape = (input_shape[0], input_shape[1], 1)

        for dim_idx, dim in enumerate(input_shape):
            if dim <= 0:
                raise ValueError("The dimension size of the inputs cannot be <= 0. Please pass a valid value to the 'input_size' parameter.")

        self.input_shape = input_shape # Shape of the input sample.
        self.layer_output_size = input_shape # Shape of the output from the current layer. For an input layer, it is the same as the shape of the input sample.

    
def single_conv(a_slice_X , W , b):
        
       
    S  = np.multiply(a_slice_X , W)
    Z  = np.sum(S)
    Z  = Z + b.astype(float)
    return Z

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
        self.initial_weights = numpy.random.randn(self.filter_bank_size[0] ,
                                                 self.filter_bank_size[1] ,
                                                 self.filter_bank_size[2] ,
                                                 self.filter_bank_size[3]) * 0.01

        # The trained filters of the conv layer. Only assigned a value after the network is trained (i.e. the train_network() function completes).
        # Just initialized to be equal to the initial filters
        self.trained_weights = self.initial_weights.copy()

        # Size of the input to the layer.
        
        self.initial_bias = numpy.zeros((1 , 1 , 1 , self.num_filters ))

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

        result = numpy.zeros(( m , n_H , n_W , N_C))
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

                        
                        result[i , h , w , c] = conv_single( A_slice ,
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
        self.dW = numpy.zeros((f , f , n_CPrev , n_C ))
        self.db = numpy.zeros((1 , 1 , 1 ,n_C))
        dAprev  = numpy.zeros((m , n_HPrev , n_WPrev , n_CPrev))
        
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


class MaxPooling2D:

    """
    Similar to the AveragePooling2D class except that it implements max pooling.
    """

    def __init__(self, pool_size, previous_layer, stride=2):
        
        """
        pool_size: Pool size.
        previous_layer: Reference to the previous layer in the CNN architecture.
        stride=2: Stride
        """
        
        if not (type(pool_size) is int):
            raise ValueError("The expected type of the pool_size is int but {pool_size_type} found.".format(pool_size_type=type(pool_size)))

        if pool_size <= 0:
            raise ValueError("The passed value to the pool_size parameter cannot be <= 0.")
        self.pool_size = pool_size

        if stride <= 0:
            raise ValueError("The passed value to the stride parameter cannot be <= 0.")
        self.stride = stride

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer

        # Size of the input to the layer.
        self.layer_input_size = self.previous_layer.layer_output_size

        # Size of the output from the layer.
        self.layer_output_size = (self.previous_layer.layer_output_size[0], numpy.uint16((self.previous_layer.layer_output_size[1] - self.pool_size )/stride + 1), 
                                  numpy.uint16((self.previous_layer.layer_output_size[2] - self.pool_size )/stride + 1), 
                                  self.previous_layer.layer_output_size[-1])

        # The layer_output attribute holds the latest output from the layer.
        self.layer_output = None

    def create_mask_from_window(self , x):
        """
        craete mask which all valus are zero except the max
        x : ths slice of the image
        """
        mask = x == numpy.max(x)
        return mask
    
    def max_pooling_back(self , dA):
        """
        get the gradints of the previous layer
        dA : the gradints from the next layer
        """
        m , n_H , n_W , n_C = dA.shape 
        
        m , n_HPrev , n_WPrev , n_CPrev  = self.input2D.shape

        dAprev = numpy.zeros((m , n_HPrev , n_WPrev , n_CPrev))
        
        f = self.pool_size

        for i in range(m):
            A = self.input2D[i]
            for h in range(n_H):                   
                    
                    for w in range(n_W):               
                        
                        for c in range(n_C):           
                            
                            vert_start = h * self.stride
                            vert_end = h * self.stride + f
                            horiz_start = w * self.stride
                            horiz_end = w * self.stride + f
                            a_prev_slice = A[vert_start:vert_end, horiz_start:horiz_end, c]
                            mask = self.create_mask_from_window(a_prev_slice)
                            dAprev[i, vert_start:vert_end, horiz_start:horiz_end, c] += numpy.multiply(mask, dA[i, h, w, c])
        return dAprev

    def max_pooling(self, input2D):
        
        """
        Applies the max pooling operation.
        
        input2D: The input to which the max pooling operation is applied.
        The max_pooling() method saves its result in the layer_output attribute.
        """
        m , n_HPrev , n_WPrev , n_CPrev = input2D.shape 
        n_W = int((n_WPrev - self.pool_size) / self.stride + 1)   
        n_H = int((n_HPrev - self.pool_size) / self.stride + 1)
        pool_out = numpy.zeros(( m , n_H , n_W , n_CPrev ))
        for i in range(m):
            Aprev = input2D[i , : , : , :]
            # Loop through the coulmns
            for h in range(n_H):
                vert_start  = h * self.stride 
                vert_end    = h * self.stride + self.pool_size
                # Loop through the rows of the image
                for w in range (n_W):
                    horiz_start = w * self.stride
                    horiz_end   = w * self.stride + self.pool_size
                    # Loop through the num of filters
                    for c in range(n_CPrev):        
                        A_slice = Aprev[vert_start : vert_end , horiz_start : horiz_end , :]
                        
                        pool_out[i , h , w , c] = numpy.max( A_slice )

        # Preparing the output of the pooling operation.
            
        self.layer_output = pool_out
        self.input2D = input2D

class Flatten:

    """
    Implementing the flatten layer.
    """

    def __init__(self, previous_layer):
        
        """
        previous_layer: Reference to the previous layer.
        """

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer

        # Size of the input to the layer.
        self.layer_input_size = self.previous_layer.layer_output_size

        # Size of the output from the layer.
        self.layer_output_size = [self.layer_input_size[1] * self.layer_input_size[2] * self.layer_input_size[3] , self.layer_input_size[0] ]
        # The layer_output attribute holds the latest output from the layer.
        self.layer_output = None

    def flatten(self, input2D):
        
        """
        Reshapes the input into a 1D vector.
        
        input2D: The input to the Flatten layer that will be converted into a 1D vector.
        The flatten() method saves its result in the layer_output attribute.
        """
        self.layer_output = flatten(input2D)
        self.layer_output_size = self.layer_output.shape
    def flatten_back(self , dA):
        """
        reshape the flatten to a size of privous shape
        dA : the gradiant of the next layer
        """
        dA = dA.reshape(self.layer_input_size)
        return dA 

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
        self.initial_weights = numpy.random.randn(self.num_neurons , self.previous_layer.layer_output_size[0] 
                                                  ) * numpy.sqrt(2 / (self.previous_layer.layer_output_size[0] + self.previous_layer.layer_output_size[0] ))
        self.initial_bias  = numpy.zeros((self.num_neurons , 1)) 
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

class GradientDescent:
    """
    Apply Gradint Descent algorithm with momuntum
    """
    def __init__(self , C  , learning_rate  , network_layers):
        """
        C = 0 : refer to momuntum
        learning_rate = 0.01 : learning rate which used in learning process
        """
        if not (C >= 0):
         raise ValueError("momuntum value cant be negative , please set positive value or zero ") 
        self.C = C
        # Set the array of the layer refrensess
        self.network_layers = network_layers
        # learning rate
        self.learning_rate = learning_rate
        # the first vx should be equal to dW
        self.vxW = [0 for i in range(len(self.network_layers))]
        self.vxb = [0 for i in range(len(self.network_layers))]

    def Learning( self ):
        for index , layer in enumerate(self.network_layers):
            if "trained_weights" in vars(layer).keys():

                if type(layer) == Dense:
                    self.vxW[index] = self.C * self.vxW[index] +  self.learning_rate * layer.dW
                    self.vxb[index] = self.C * self.vxb[index] +  self.learning_rate * layer.db

                    layer.trained_weights = layer.trained_weights +  self.vxW[index]
                    layer.trained_bias    = layer.trained_bias    +  self.vxb[index]
                
                else:
                    self.vxW[index] = self.C * self.vxW[index] +  self.learning_rate * layer.dW
                    layer.trained_weights = layer.trained_weights + self.vxW[index] 



class RMS:
    """
    Apply Gradint Descent algorithm with momuntum
    """
    def __init__(self  , learning_rate  , network_layers , decay_rate ):
        """
        C = 0 : refer to momuntum
        learning_rate = 0.01 : learning rate which used in learning process
        """
        self.decay_rate = decay_rate
        # Set the array of the layer refrensess
        self.network_layers = network_layers
        # learning rate
        self.learning_rate = learning_rate
        # the first vx should be equal to dW
        self.grad_squard_W = [0 for i in range(len(self.network_layers))]
        self.grad_squard_b = [0 for i in range(len(self.network_layers))]

    def Learning( self ):
        for index , layer in enumerate(self.network_layers):
            if "trained_weights" in vars(layer).keys():

                if type(layer) == Dense:
                    self.grad_squard_W[index] = self.decay_rate * self.grad_squard_W[index] + (1 - self.decay_rate) * layer.dW * layer.dW
                    self.grad_squard_b[index] = self.decay_rate * self.grad_squard_b[index] + (1 - self.decay_rate) * layer.db * layer.db

                    const_W = np.sqrt(self.grad_squard_W[index]) + 0.000001 
                    const_b = np.sqrt(self.grad_squard_b[index]) + 0.000001 

                    layer.trained_weights -= (self.learning_rate * layer.trained_weights / const_W) * layer.dW  
                    layer.trained_bias    -= (self.learning_rate * layer.trained_bias / const_b)   * layer.db
                
                else:
                    self.grad_squard_W[index] = self.decay_rate * self.grad_squard_W[index] + (1 - self.decay_rate) * layer.dW * layer.dW
                    const_W = np.sqrt(self.grad_squard_W[index]) + 0.000001 

                    layer.trained_weights -= (self.learning_rate * layer.trained_weights / const_W ) * layer.dW


class Adam:
    """
    Apply Gradint Descent algorithm with momuntum
    """
    def __init__(self  , beta1 , beta2 , learning_rate  , network_layers ):
        """
        C = 0 : refer to momuntum
        learning_rate = 0.01 : learning rate which used in learning process
        """
        self.beta1 = beta1
        self.beta2 = beta2
        # Set the array of the layer refrensess
        self.network_layers = network_layers
        # learning rate
        self.learning_rate = learning_rate
        # the first momuntum
        self.FirstMW  =  [0 for i in range(len(self.network_layers))]
        self.FirstMb  =  [0 for i in range(len(self.network_layers))]
        # the second Mommuntum
        self.SecondMW =  [0 for i in range(len(self.network_layers))]
        self.SecondMb =  [0 for i in range(len(self.network_layers))]


    def Learning( self ):
        for index , layer in enumerate(self.network_layers):
            if "trained_weights" in vars(layer).keys():

                self.FirstMW[index]  = self.beta1 * self.FirstMW[index]  + (1 - self.beta1) * layer.dW 
                self.SecondMW[index] = self.beta2 * self.SecondMW[index] + (1 - self.beta1) * layer.dW  * layer.dW

                const_W = np.sqrt(self.SecondMW[index]) + 0.000001

                self.FirstMb[index]  = self.beta1 * self.FirstMb[index]  + (1 - self.beta1) * layer.db 
                self.SecondMb[index] = self.beta2 * self.SecondMb[index] + (1 - self.beta1) * layer.db  * layer.db
                    
                const_b = np.sqrt(self.SecondMb[index]) + 0.000001 

                layer.trained_weights -= (self.learning_rate *  self.FirstMW[index] / const_W )  * layer.dW
                layer.trained_bias    -= (self.learning_rate *  self.FirstMb[index] / const_b )  * layer.db
               



class AdaGrad:
    """
    Apply Gradint Descent algorithm with momuntum
    """
    def __init__(self  , learning_rate  , network_layers):
        """
        C = 0 : refer to momuntum
        learning_rate = 0.01 : learning rate which used in learning process
        """
        # Set the array of the layer refrensess
        self.network_layers = network_layers
        # learning rate
        self.learning_rate = learning_rate
        # the first vx should be equal to dW
        self.grad_squard_W = [0 for i in range(len(self.network_layers))]
        self.grad_squard_b = [0 for i in range(len(self.network_layers))]

    def Learning( self ):
        for index , layer in enumerate(self.network_layers):
            if "trained_weights" in vars(layer).keys():

                if type(layer) == Dense:
                    self.grad_squard_W[index] += layer.dW * layer.dW
                    self.grad_squard_b[index] += layer.db * layer.db

                    const_W = np.sqrt(self.grad_squard_W[index]) + 0.000001 
                    const_b = np.sqrt(self.grad_squard_b[index]) + 0.000001 

                    layer.trained_weights -= (self.learning_rate * layer.trained_weights / const_W ) * layer.dW
                    layer.trained_bias    -= (self.learning_rate * layer.trained_bias / const_b )    * layer.db
                
                else:
                    self.grad_squard_W[index] = layer.dW * layer.dW
                    const_W = np.sqrt(self.grad_squard_W[index]) + 0.000001 

                    layer.trained_weights -= (self.learning_rate * layer.trained_weights / const_W ) * layer.dW


class Model:

    """
    Creating a CNN model.
    """

    def __init__(self, last_layer, epochs=10, learning_rate=0.01 , learning_algorithm = "AdaGrad"):
        
        """
        last_layer: A reference to the last layer in the CNN architecture.
        epochs=10: Number of epochs.
        learning_rate=0.01: Learning rate.
        """
        if learning_algorithm == "AdaGrad":
            self.learning = AdaGrad
        
        elif learning_algorithm == "GradientDescent":
            self.learning = GradientDescent
        
        elif learning_algorithm == "RMS":
            self.learning = RMS
        
        elif learning_algorithm == "Adam":
            self.learning = Adam
        
        else:
            raise ValueError("{learning_algorithm} if not defined yet ".format(learning_algorithm = learning_algorithm))
        
        self.last_layer = last_layer
        self.epochs = epochs
        self.learning_rate = learning_rate

        # The network_layers attribute is a list holding references to all CNN layers.
        self.network_layers = self.get_layers()

    def get_layers(self):

        """
        Prepares a  list of all layers in the CNN model.
        Returns the list.
        """

        network_layers = []

        # The last layer in the network archietcture.
        layer = self.last_layer

        while "previous_layer" in layer.__init__.__code__.co_varnames:
            network_layers.insert(0, layer)
            layer = layer.previous_layer

        return network_layers

    def train(self, train_inputs, train_outputs , 
            batch_size , validation_set = None ,
             C = 0.6 , decay_rate = 0.6 , beta1 = 0.9 , beta2 = 0.9):
        
        """
        Trains the CNN model.
        It is important to note that no learning algorithm is used for training the CNN. Just the learning rate is used for making some changes which is better than leaving the weights unchanged.
        
        train_inputs: Training data inputs.
        train_outputs: Training data outputs.
        batch_size : size of the sample in every feed process 
        """
        self.history = []
        if (train_inputs.ndim != 4):
            raise ValueError("The training data input has {num_dims} but it must have 4 dimensions. The first dimension is the number of training samples, the second & third dimensions represent the width and height of the sample, and the fourth dimension represents the number of channels in the sample.".format(num_dims=train_inputs.ndim))    
        
        if batch_size > train_inputs.shape[0]:
            raise ValueError("Batch size {batch_size}cant be greater than the numper of samples {sample_size}".format(batch_size = batch_size , sample_size = train_inputs.shape[0]))
        self.batch_size = batch_size

        Batches = train_inputs.shape[0] // batch_size

        if self.learning == AdaGrad:
            self.update_method = AdaGrad(learning_rate  = self.learning_rate ,
                                                         network_layers =  self.network_layers )
        
        elif self.learning == GradientDescent:
            self.update_method = GradientDescent(C = C ,
                                                        learning_rate  = self.learning_rate , 
                                                        network_layers =  self.network_layers )
        
        elif self.learning == RMS:
            self.update_method = RMS(decay_rate = decay_rate ,
                                                        learning_rate  = self.learning_rate , 
                                                        network_layers =  self.network_layers )

        elif self.learning == Adam:
            self.update_method = Adam(beta1 = beta1 , beta2 = beta2 ,
                                                        learning_rate  = self.learning_rate , 
                                                        network_layers =  self.network_layers)

        for epoch in range(self.epochs):
            acclist  = []
            costlist = []
            for batch in range(Batches):
                #apply forward propagation
                first_index  = batch * self.batch_size
                second_index = (batch + 1 ) * self.batch_size
                X_batch = train_inputs[first_index : second_index , :]
                
                predicted_label = self.feed_sample(X_batch)
                y_batch = train_outputs[first_index : second_index]
                
                cost , dZ , acc = categorical_crossentropy(predicted_label , y_batch)
                costlist.append(cost)
                acclist.append(acc)
                self.feed_back(dZ)
                self.update_method.Learning()
            cost = numpy.mean(costlist)
            acc  = numpy.mean(acclist )
            self.history.append([cost , acc])
            print("Epoch {epoch} cost = {cost} accuracy = {acc}".format(epoch = epoch +  1 , cost = cost ,  acc = acc))



    def feed_sample(self, sample):
        
        """
        Feeds a sample in the CNN layers.
        
        sample: The samples to be fed to the CNN layers.
        
        Returns results of the last layer in the CNN.
        """

        last_layer_outputs = sample
        for layer in self.network_layers:
            if type(layer) is Conv2D:
                layer.conv(input2D=last_layer_outputs)
            elif type(layer) is Dense:
                layer.dense_layer(layer_input=last_layer_outputs)
            elif type(layer) is MaxPooling2D:
                layer.max_pooling(input2D=last_layer_outputs)
            elif type(layer) is Flatten:
                layer.flatten(input2D=last_layer_outputs)
            elif type(layer) is Input2D:
                pass
            else:
                print("Other")
                raise TypeError("The layer of type {layer_type} is not supported yet.".format(layer_type=type(layer)))

            last_layer_outputs = layer.layer_output
        return self.network_layers[-1].layer_output
    
    def feed_back(self , dZ):
        last_layer_derivative = dZ
        Last = True
        for layer in range(1 , len(self.network_layers) +1):
            layer = self.network_layers[-layer]
            if type(layer) is Conv2D:
                last_layer_derivative = layer.conv_back(dA = last_layer_derivative)
            elif type(layer) is Dense:
                last_layer_derivative = layer.dense_layer_back(dA = last_layer_derivative , Last = Last)
            elif type(layer) is MaxPooling2D:
                last_layer_derivative = layer.max_pooling_back(dA = last_layer_derivative)
            elif type(layer) is Flatten:
                last_layer_derivative = layer.flatten_back(dA = last_layer_derivative)
            elif type(layer) is Input2D:
                pass
            else:
                print("Other")
                raise TypeError("The layer of type {layer_type} is not supported yet.".format(layer_type=type(layer)))
            Last = False

     
    def update_weights(self , C = 0):
        
        """
        Updates the weights of the CNN.
        It is important to note that no learning algorithm is used for training the CNN. Just the learning rate is used for making some changes which is better than leaving the weights unchanged.
        
        This method loops through the layers and updates their weights.
        network_error: The network error in the last epoch.
        """
        
        for layer in self.network_layers:
            if "trained_weights" in vars(layer).keys():
                layer.trained_weights = layer.trained_weights - self.learning_rate * layer.dW # here you have to add derivative
                layer.trained_bias    = layer.trained_bias    - self.learning_rate * layer.db

    def predict(self, data_inputs):

        """
        Uses the trained CNN for making predictions.
        
        data_inputs: The inputs to predict their label.
        Returns a list holding the samples predictions.
        """

        if (data_inputs.ndim != 4):
            raise ValueError("The data input has {num_dims} but it must have 4 dimensions. The first dimension is the number of training samples, the second & third dimensions represent the width and height of the sample, and the fourth dimension represents the number of channels in the sample.".format(num_dims=data_inputs.ndim))

        predictions = []
        for sample in data_inputs:
            probs = self.feed_sample(sample=sample)
            predicted_label = numpy.where(numpy.max(probs) == probs)[0][0]
            predictions.append(predicted_label)
        return predictions

    def summary(self):

        """
        Prints a summary of the CNN architecture.
        """

        print("\n----------Network Architecture----------")
        for layer in self.network_layers:
            print(type(layer))
        print("----------------------------------------\n")

def flatten(Matrix):
    shapes  = Matrix.shape 
    Updates = np.zeros((shapes[1]*shapes[2]*shapes[3] , shapes[0]))
    for sample in range(shapes[0]):
        Updates[: , sample] = np.ravel(Matrix[sample])
    return Updates
