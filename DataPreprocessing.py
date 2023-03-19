import numpy as np
import keras
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessing:
    def dataloading(self):
        (X_train, Y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, Y_train, stratify=Y_train, random_state=7, test_size=0.1
        )
        X_train_scaled = X_train.reshape(len(X_train), 28 * 28) / 255.0
        X_val_scaled = X_val.reshape(len(X_val), 28 * 28) / 255.0
        X_test_scaled = X_test.reshape(len(X_test), 28 * 28) / 255.0

        enc = OneHotEncoder()
        y_train_enc = enc.fit_transform(np.expand_dims(y_train, 1)).toarray()
        y_val_enc = enc.fit_transform(np.expand_dims(y_val, 1)).toarray()
        y_test_enc = enc.fit_transform(np.expand_dims(y_test, 1)).toarray()
        # print(y_train_enc.shape, y_val_enc.shape, y_test_enc.shape)

        return (
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            y_train_enc,
            y_val_enc,
            y_test_enc,
            y_val,
            y_test,
        )
