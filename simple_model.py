from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU, regularizers, Activation, Flatten
from keras.callbacks import ModelCheckpoint


class SimpleModel:
    def fit(self, X_train, y_train, model_file="simple"):
        self.X_train = X_train
        self.y_train = y_train

        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dense(16))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.add(Activation('linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        checkpointer = ModelCheckpoint(filepath=model_file, verbose=1, save_best_only=True)
        self.history = self.model.fit(X_train, y_train, epochs=10, batch_size=128, callbacks=[checkpointer],
                                      shuffle=True)

    def predict(self, X_test):
        return self.model.predict(X_test)
