# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
 
class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        # initialize the model
        print((depth, height, width))
        model = Sequential()


        # first set of CONV => RELU => POOL
        model.add(Convolution2D(30, 5, 5, activation="relu", border_mode="full", input_shape=(depth, height, width)))
        model.add(Dropout(0.3))
        #model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))

        # second set of CONV => RELU => POOL
        model.add(Convolution2D(80, 5, 5, activation="relu", border_mode="full"))
        model.add(Dropout(0.3))
        #model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))

        # third set of CONV => RELU => POOL
        model.add(Convolution2D(80, 5, 5, activation="relu", border_mode="full"))
        model.add(Dropout(0.3))
        #model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
 
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # return the constructed network architecture
        return model