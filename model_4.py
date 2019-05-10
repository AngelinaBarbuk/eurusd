from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU, regularizers, Activation, Flatten
from keras.callbacks import ModelCheckpoint
from tensorflow.python.estimator import keras

from base_model import BaseModel
from utilities import get_path


class Model_4(BaseModel):
    def __init__(self, file=None):
        if file:
            self.model = keras.models.load_model(file)

    def create(self, shape=[60, 4]):
        self.model = Sequential()
        self.model.add(Dense(256, input_shape=shape))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dense(64))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dense(16))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.add(Activation('linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X_train, y_train, X_valid, y_valid, model_file="models/model_4.h5"):
        super(Model_4, self).fit(X_train, y_train, X_valid, y_valid, model_file)
