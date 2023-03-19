import numpy as np
import keras
import wandb
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

wandb.login()


class DataPreprocessing:
    def dataloading(self):
        (X_train, Y_train), (X_test, y_test) = mnist.load_data()
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

        wandb.init(project="CS6910_Assignment1")
        class_labels = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot",
        }
        unique, indices = np.unique(y_train, return_index=True)
        images = X_train_scaled[indices]
        labels = y_train[indices]
        images = images.reshape(-1, 28, 28, 1)
        grid = wandb.Image(
            np.concatenate(images, axis=1),
            caption=[class_labels[label] for label in labels],
        )

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
