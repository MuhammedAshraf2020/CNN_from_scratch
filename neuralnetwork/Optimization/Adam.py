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
