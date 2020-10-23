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
