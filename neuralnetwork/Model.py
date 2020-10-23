import numpy as np
from neuralnetwork.utils import categorical_crossentropy
from neuralnetwork.optimizers.RMS import RMS
from neuralnetwork.optimizers.Adam import Adam
from neuralnetwork.optimizers.GradientDescent import GradientDescent
from neuralnetwork.optimizers.AdaGrad import AdaGrad
from neuralnetwork.layers import Conv2D , MaxPooling2D , Dense , Flatten


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
