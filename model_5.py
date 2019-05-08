from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU, regularizers, Activation, Flatten, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from tensorflow.python.estimator import keras


class Model_5:
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

    def fit(self, X_train, y_train, model_file="models/model_5.h5"):
        self.X_train = X_train
        self.y_train = y_train
        checkpointer = ModelCheckpoint(filepath="checkpoints/" + model_file, verbose=1, save_best_only=True)
        self.history = self.model.fit(X_train, y_train, epochs=10, batch_size=128, callbacks=[checkpointer],
                                      shuffle=True)
        self.model.save(model_file)

    def predict(self, X_test):
        return self.model.predict(X_test)
