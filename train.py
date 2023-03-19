import numpy as np
from sklearn.preprocessing import StandardScaler
import ModelFinal
import DataPreprocessing
import wandb

# API = 887362a2ceb2116d60b2d826763161b8361e55a1
wandb.login()

# import argparse
# parser = argparse.ArgumentParser()

# parser.add_argument('-wp', '--wandb_project', default = 'myprojectname')
# parser.add_argument('-we', '--wandb_entity', default = 'myname')
# parser.add_argument('-d', '--dataset', default='fashion_mnist')
# parser.add_argument('-e', '--epochs', default = 5, type= int)
# parser.add_argument('-b', '--batch_size', default = 4, type= int)
# parser.add_argument('-o', '--optimizer', default = 'ADAM', choices = ['SGD', 'MGD', 'NAG', 'RMSPROP', 'ADAM', 'NADAM'], type = str)
# parser.add_argument('-lr', '--learning_rate', default = 1e-4, type= float)
# parser.add_argument('-w_d', '--weight_decay', default = 0.0, type= float)
# parser.add_argument('-w_i', '--weight_init', default = 'RANDOM', type= str)
# parser.add_argument('-nhl', '--num_layers', default = 1, type= int)
# parser.add_argument('-sz', '--hidden_size', default = 32, type = int)
# parser.add_argument('-a', '--activation', default = 'TANH', type = str)

# args = parser.parse_args()

model = ModelFinal.Model()
data = DataPreprocessing.DataPreprocessing()
(
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    y_train_enc,
    y_val_enc,
    y_test_enc,
    y_val,
    y_test,
) = data.dataloading()

# (X_train, Y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, Y_train, stratify=Y_train, random_state=7, test_size=0.1
# )
# X_train_scaled = X_train.reshape(len(X_train), 28 * 28) / 255.0
# X_val_scaled = X_val.reshape(len(X_val), 28 * 28) / 255.0
# X_test_scaled = X_test.reshape(len(X_test), 28 * 28) / 255.0

# enc = OneHotEncoder()
# y_train_enc = enc.fit_transform(np.expand_dims(y_train, 1)).toarray()
# y_val_enc = enc.fit_transform(np.expand_dims(y_val, 1)).toarray()
# y_test_enc = enc.fit_transform(np.expand_dims(y_test, 1)).toarray()
# print(y_train_enc.shape, y_val_enc.shape, y_test_enc.shape)


# model.fit(X = X_train_scaled,
#           Y = y_train_enc,
#           X_val = X_val_scaled,
#           Y_val = y_val,
#           epochs = args.epochs,
#           learning_rate = args.learning_rate,
#           weight_decay = args.weight_decay,
#           hidden_layers = args.num_layers,
#           neurons_per_layer = args.hidden_size,
#           batch_size = args.batch_size,
#           optimizer = args.optimizer,
#           initialization_alg = args.weight_init,
#           activation_function = args.activation)

# model.fit(
#     X=X_train_scaled,
#     Y=y_train_enc,
#     X_val=X_val_scaled,
#     Y_val=y_val,
#     Y_val_enc=y_val_enc,
#     epochs=75,
#     learning_rate=1e-7,
#     weight_decay=0.0005,
#     hidden_layers=1,
#     neurons_per_layer=32,
#     batch_size=16,
#     optimizer="NADAM",
#     initialization_alg="XAVIER",
#     activation_function="RELU",
# )
#### Sweep configuration ####
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "loss_epoch"},
    "parameters": {
        "epochs": {"values": [20]},
        "num_layers": {"values": [1, 2]},
        "layer_size": {"values": [32, 64, 128]},
        "learning_rate": {"values": [1e-4, 1e-5, 5e-6]},
        "weight_decay": {"values": [0, 0.0005, 0.05]},
        "optimizer": {"values": ["SGD", "MGD", "NAG", "RMSPROP", "ADAM", "NADAM"]},
        "batch_size": {"values": [16, 32]},
        "initialization": {"values": ["RANDOM", "XAVIER", "HE"]},
        "activation_function": {"values": ["SIGMOID", "TANH"]},
    },
}


def wandbsweeps():
    wandb.init(project="CS6910_Assignment1")
    wandb.run.name = (
        "lr_"
        + str(wandb.config.learning_rate)
        + "opt"
        + str(wandb.config.optimizer)
        + "epoch"
        + str(wandb.config.epochs)
        + "bs"
        + str(wandb.config.batch_size)
        + "act"
        + str(wandb.config.activation_function)
    )
    model.fit(
        X=X_train_scaled,
        Y=y_train_enc,
        X_val=X_val_scaled,
        Y_val=y_val,
        Y_val_enc=y_val_enc,
        epochs=wandb.config.epochs,
        learning_rate=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
        hidden_layers=wandb.config.num_layers,
        neurons_per_layer=wandb.config.layer_size,
        batch_size=wandb.config.batch_size,
        optimizer=wandb.config.optimizer,
        initialization_alg=wandb.config.initialization,
        activation_function=wandb.config.activation_function,
    )


sweep_id = wandb.sweep(sweep=sweep_configuration, project="CS6910_Assignment1")
wandb.agent(sweep_id, function=wandbsweeps, count=3)
