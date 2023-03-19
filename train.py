import ModelFinal
import DataPreprocessing
import wandb

wandb.login()

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-wp", "--wandb_project", default="myprojectname")
parser.add_argument("-we", "--wandb_entity", default="myname")
parser.add_argument("-d", "--dataset", default="fashion_mnist")
parser.add_argument("-e", "--epochs", default=5, type=int)
parser.add_argument("-b", "--batch_size", default=4, type=int)
parser.add_argument(
    "-o",
    "--optimizer",
    default="ADAM",
    choices=["SGD", "MGD", "NAG", "RMSPROP", "ADAM", "NADAM"],
    type=str,
)
parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
parser.add_argument("-w_d", "--weight_decay", default=0.0, type=float)
parser.add_argument("-w_i", "--weight_init", default="RANDOM", type=str)
parser.add_argument("-nhl", "--num_layers", default=1, type=int)
parser.add_argument("-sz", "--hidden_size", default=32, type=int)
parser.add_argument("-a", "--activation", default="TANH", type=str)
parser.add_argument("-l", "--loss", default="CE", type=str)

args = parser.parse_args()

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

model.fit(
    X=X_train_scaled,
    Y=y_train_enc,
    X_val=X_val_scaled,
    Y_val=y_val,
    epochs=args.epochs,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    hidden_layers=args.num_layers,
    neurons_per_layer=args.hidden_size,
    batch_size=args.batch_size,
    optimizer=args.optimizer,
    initialization_alg=args.weight_init,
    activation_function=args.activation,
    loss_function=args.loss_function,
)

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
#      loss_function = "CE"
# )
#### Sweep configuration ####
sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "loss_epoch"},
    "parameters": {
        "epochs": {"values": [20]},
        "num_layers": {"values": [3]},
        "layer_size": {"values": [64]},
        "learning_rate": {"values": [1e-5]},
        "weight_decay": {"values": [0.005]},
        "optimizer": {"values": ["NADAM"]},
        "batch_size": {"values": [8]},
        "initialization": {"values": ["XAVIER"]},
        "activation_function": {"values": ["TANH"]},
        "loss_function": {"values": ["CE"]},
        # "dataset": {"values": ["FASHION_MNIST"]},
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
        X_test=X_test_scaled,
        Y_test=y_test,
        epochs=wandb.config.epochs,
        learning_rate=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
        hidden_layers=wandb.config.num_layers,
        neurons_per_layer=wandb.config.layer_size,
        batch_size=wandb.config.batch_size,
        optimizer=wandb.config.optimizer,
        initialization_alg=wandb.config.initialization,
        activation_function=wandb.config.activation_function,
        loss_function=wandb.config.loss_function,
    )


sweep_id = wandb.sweep(sweep=sweep_configuration, project="CS6910_Assignment1")
wandb.agent(sweep_id, function=wandbsweeps)
# wandb.agent(sweep_id, function=wandbsweeps, count=3)
