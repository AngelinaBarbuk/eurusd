from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU, regularizers, Activation, Flatten, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from tensorflow.python.estimator import keras

from base_model import BaseModel


class Model_5(BaseModel):
    def __init__(self, file=None):
        if file:
            self.model = keras.models.load_model(file)

    def create(self, shape=[60, 4]):
        self.model = Sequential()
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(LSTM(units=256, return_sequences=True, input_shape=shape))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(LSTM(units=64, return_sequences=True, input_shape=shape))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(LSTM(units=1, input_shape=shape))
        self.model.add(Activation('linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X_train, y_train, X_valid, y_valid, model_file="models/model_5.h5"):
        super(Model_5, self).fit(X_train, y_train, X_valid, y_valid, model_file)
