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

def single_conv(a_slice_X , W , b):


    S  = np.multiply(a_slice_X , W)
    Z  = np.sum(S)
    Z  = Z + b.astype(float)
    return Z

def flatten(Matrix):
    shapes  = Matrix.shape
    Updates = np.zeros((shapes[1]*shapes[2]*shapes[3] , shapes[0]))
    for sample in range(shapes[0]):
        Updates[: , sample] = np.ravel(Matrix[sample])
    return Updates

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
