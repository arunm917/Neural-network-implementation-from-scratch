import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm.notebook import trange, tqdm
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from sklearn.metrics import accuracy_score
import Model as model
import wandb
import random

# wandb.login()

(X_train, Y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
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
print(y_train_enc.shape, y_val_enc.shape, y_test_enc.shape)

model.fit(
    X=X_train_scaled,
    Y=y_train_enc,
    X_val=X_val_scaled,
    Y_val=y_val,
    epochs=20,
    learning_rate=1e-6,
    weight_decay=1e-6,
    hidden_layers=1,
    neurons_per_layer=64,
    batch_size=32,
    optimizer="ADAM",
    initialization_alg="XAVIER",
    activation_function="RELU",
)
