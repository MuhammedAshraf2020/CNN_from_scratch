from neuralnetwork.utils import flatten
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
