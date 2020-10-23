import numpy as np
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
        self.layer_output_size = (self.previous_layer.layer_output_size[0], np.uint16((self.previous_layer.layer_output_size[1] - self.pool_size )/stride + 1),
                                  np.uint16((self.previous_layer.layer_output_size[2] - self.pool_size )/stride + 1),
                                  self.previous_layer.layer_output_size[-1])

        # The layer_output attribute holds the latest output from the layer.
        self.layer_output = None

    def create_mask_from_window(self , x):
        """
        craete mask which all valus are zero except the max
        x : ths slice of the image
        """
        mask = x == np.max(x)
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
