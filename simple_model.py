from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU, regularizers, Activation, Flatten
from keras.callbacks import ModelCheckpoint
from tensorflow.python.estimator import keras


class SimpleModel:
    def __init__(self, file=None):
        if file:
            self.model = keras.models.load_model(file)

    def create(self, shape=[60, 4]):
        self.model = Sequential()
        self.model.add(Dense(64, input_shape=shape))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dense(16))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.add(Activation('linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X_train, y_train, model_file="simple"):
        self.X_train = X_train
        self.y_train = y_train
        checkpointer = ModelCheckpoint(filepath=model_file, verbose=1, save_best_only=True)
        self.history = self.model.fit(X_train, y_train, epochs=10, batch_size=128, callbacks=[checkpointer],
                                      shuffle=True)
        self.model.save('models/simple.h5')

    def predict(self, X_test):
        return self.model.predict(X_test)
