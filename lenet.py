# import the necessary packages
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
import numpy as np


class ModelWrapper:

    def __init__(self, models, bag_size) -> None:
        super().__init__()
        self.models = models
        self.bag_size = bag_size

    @staticmethod
    def bag(train_x, train_y, num_bags, bag_size):
        bags_x, bags_y = [], []
        for _ in range(num_bags):
            indices = np.random.permutation(np.arange(train_x.shape[0]))[:bag_size]
            bags_x.append(train_x[indices])
            bags_y.append(train_y[indices])
        return bags_x, bags_y

    def compile(self, **params):
        for model in self.models:
            model.compile(**params)

    def fit(self, train_x, train_y, **params):
        num_bags = len(self.models)
        bags_x, bags_y = ModelWrapper.bag(train_x, train_y, num_bags, self.bag_size)
        for i in range(num_bags):
            print("Fitting model " + str(i + 1) + " of " + str(num_bags))
            self.models[i].fit(bags_x[i], bags_y[i], **params)

    def predict(self, test_x, **params):
        num_bags = len(self.models)
        predictions = np.zeros((num_bags, test_x.shape[0], 2))
        average = np.zeros((test_x.shape[0], 2))
        for i in range(num_bags):
            predictions[i, :, :] = self.models[i].predict(test_x, **params)
        average[:, 0] = np.sum(predictions[:, :, 0], axis=0) / num_bags
        average[:, 1] = 1 - average[:, 0]
        return average

    def predict_single(self, x, threshold, **params):
        test_x = np.zeros((1, x.shape[0], x.shape[1], 1))
        test_x[0, :, :, 0] = x
        prediction = self.predict(test_x, **params)[0][1]
        return prediction > threshold

    def save(self):
        jsons = [model.to_json() for model in self.models]
        for i in range(len(jsons)):
            with open("network_" + str(i) + ".json", "w") as json_file:
                json_file.write(jsons[i])
            self.models[i].save_weights("weights_" + str(i) + ".h")

    def load(self, num_bags):
        for i in range(num_bags):
            try:
                json_file = open("network_" + str(i) + ".json", 'r')
            except IOError:
                return False
            json = json_file.read()
            json_file.close()
            model = model_from_json(json)
            model.load_weights("weights_" + str(i) + ".h")
            self.models.append(model)



class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        '''
        initialize the model
        '''
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using 'channels first', update the input shape
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)

        # first set of CONV => ReLU => POOL layers
        model.add(Conv2D(20, (5, 5), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second layer of CONV => ReLU => POOL layers
        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => ReLU layeres
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model

    @staticmethod
    def build_bagged(num_bags, bag_size, **params):
        models = []
        for i in range(num_bags):
            models.append(LeNet.build(**params))
        return ModelWrapper(models, bag_size)

    @staticmethod
    def load_bagged(num_bags):
        model = ModelWrapper([], 0)
        model.load(num_bags)
        return model
