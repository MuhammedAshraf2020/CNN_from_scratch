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

                self.vxW[index] = self.C * self.vxW[index] +  self.learning_rate * layer.dW
                self.vxb[index] = self.C * self.vxb[index] +  self.learning_rate * layer.db

                layer.trained_weights = layer.trained_weights +  self.vxW[index]
                layer.trained_bias    = layer.trained_bias    +  self.vxb[index]
