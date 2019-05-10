from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


class BaseModel:
    def fit(self, X_train, y_train, X_valid, y_valid, model_file):
        self.X_train = X_train
        self.y_train = y_train

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        checkpointer = ModelCheckpoint(filepath="checkpoints/" + model_file, verbose=1, save_best_only=True)
        self.history = self.model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=20, batch_size=128,
                                      callbacks=[checkpointer, es], shuffle=True)
        fig, ax = plt.subplots(figsize=(20, 17))
        ax.plot(self.history.history['loss'], color='red', label='Loss')
        ax.plot(self.history.history['val_loss'], color='blue', label='Validation Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        ax.legend()
        plt.show()
        self.model.save(model_file)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predicted = self.predict(X_test)
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[0], scores * 100))
        fig, ax = plt.subplots(figsize=(20, 17))
        ax.plot(y_test, color='red', label='Real Price')
        ax.plot(predicted, color='blue', label='Predicted Price')
        ax.get_xaxis().set_visible(False)
        plt.title('Price Prediction')
        ax.legend()
        plt.show()
