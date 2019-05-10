from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU, regularizers, Activation, Flatten, LSTM
from keras.callbacks import ModelCheckpoint
from tensorflow.python.estimator import keras

from base_model import BaseModel


class Model_3(BaseModel):
    def __init__(self, file=None):
        if file:
            self.model = keras.models.load_model(file)

    def create(self, shape=[60, 4]):
        self.model = Sequential()
        self.model.add(LSTM(units=256, input_shape=shape))
        self.model.add(Dense(1))
        self.model.add(Activation('linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X_train, y_train, X_valid, y_valid, model_file="models/model_2.h5"):
        super(Model_3, self).fit(X_train, y_train, X_valid, y_valid, model_file)
