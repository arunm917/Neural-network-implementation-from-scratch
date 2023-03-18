import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import ActivationFunctions
import LossFunctions
import ParameterInitialization

initial = ParameterInitialization.ParameterInitialization()
act = ActivationFunctions.ActivationFunctions()
loss_functions = LossFunctions.LossFunctions()


class Model:
    def forward_prop(self, x, activation_function, nh):
        self.A = {}
        self.H = {}
        self.nh = nh
        self.x = x
        self.H[0] = self.x.T

        for i in range(self.nh + 1):
            self.A[i + 1] = np.matmul(self.W[i + 1], self.H[i]) + self.B[i + 1]
            self.H[i + 1] = act.activation(self.A[i + 1], activation_function)

        self.y_pred = act.softmax(self.A[i + 1])
        # print(self.y_pred)
        # print('Y = ', self.y.shape, 'Y_pred =', self.y_pred.shape)
        return self.y_pred

    def grad(self, x, y, activation_function, W, B, nh):
        self.nh = nh
        if W is None:
            W = self.W
        if B is None:
            B = self.B

        y_pred = self.forward_prop(x, activation_function, self.nh)
        self.dW = {}
        self.dB = {}
        self.dA = {}
        self.dH = {}
        self.dA[self.nh + 1] = y_pred - y.T

        for i in range(self.nh + 1, 0, -1):
            self.dW[i] = np.matmul(self.dA[i], self.H[i - 1].T)
            self.dB[i] = np.sum(self.dA[i], axis=1).reshape(-1, 1)
            self.dH[i - 1] = np.matmul(W[i].T, self.dA[i])
            self.dA[i - 1] = np.multiply(
                self.dH[i - 1], act.der_activation(self.H[i - 1], activation_function)
            )
        return (self.dW, self.dB)

    def fit(
        self,
        X,
        Y,
        X_val,
        Y_val,
        epochs,
        learning_rate,
        weight_decay,
        hidden_layers,
        neurons_per_layer,
        batch_size,
        optimizer,
        initialization_alg,
        activation_function,
    ):

        # x = X[0:batch_size,:]
        # y = Y[0:batch_size,:]
        # self.X = X
        self.loss_epoc_store = []
        self.nx = X.shape[1]
        self.ny = Y.shape[1]
        self.nh = hidden_layers
        self.neurons = neurons_per_layer
        self.initialization_alg = initialization_alg
        # self.weight_decay = weight_decay
        hidden_layer_sizes = [self.neurons] * self.nh
        self.sizes = [self.nx] + hidden_layer_sizes + [self.ny]
        print(self.sizes)
        if optimizer == "SGD":
            batch_size = 1

        step_size = len(X) / batch_size

        self.W, self.B = initial.initialize(self.sizes, self.initialization_alg)
        opt = Optimizer(
            self.W,
            self.B,
            self.sizes,
            batch_size,
            learning_rate,
            weight_decay,
            optimizer,
            activation_function,
        )

        for i in tqdm(
            range(epochs), total=epochs, unit="epoch", leave=True, dynamic_ncols=False
        ):
            epoch = i + 1
            step = 0
            # print('epoch = ', epoch)
            loss_epoch = 0
            self.loss_batch_store = []

            for i in range(0, len(X), batch_size):
                step += 1
                # print('step = ', step)
                self.x = X[i : i + batch_size, :]
                self.y = Y[i : i + batch_size, :]
                # print('x =', self.x.shape)
                # print('y =', self.y.shape)
                (self.dW, self.dB) = self.grad(
                    self.x, self.y, activation_function, self.W, self.B, self.nh
                )
                self.W, self.B = opt.learning_algoithms(
                    self.x, self.y, self.dW, self.dB, step
                )

                # Predicting loss for each batch
                # loss_batch = loss_functions.cross_entropy(self.y.T ,self.y_pred)
                # self.loss_batch_store.append(loss_batch)
            loss_epoch = loss_functions.cross_entropy(self.y.T, self.y_pred)
            print("training loss = ", round(loss_epoch, 4))
            self.loss_epoc_store.append(loss_epoch)

            # Predicting validation loss
            y_pred_val = self.forward_prop(X_val, activation_function)
            Y_pred_val = np.array(y_pred_val.T).squeeze()
            Y_pred_val = np.argmax(Y_pred_val, 1)
            # print(Y_pred_val[0:10], Y_val[0:10])
            accuracy_val = accuracy_score(Y_pred_val, Y_val)
            print("validation accuracy = ", round(accuracy_val, 2))
            # wandb.log({'loss_epoch': loss_epoch, 'accuracy_val':accuracy_val, 'epochs':epoch})

        plt.plot(self.loss_epoc_store)
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.show()
        return (accuracy_val, loss_epoch)


model = Model()


class Optimizer:
    def __init__(
        self,
        W,
        B,
        sizes,
        batch_size,
        learning_rate,
        weight_decay,
        optimizer,
        activation_function,
    ):
        self.W = W
        self.B = B
        # self.dW = dW
        # self.dB = dB
        self.sizes = sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.activation_function = activation_function
        # self.epoch = epoch
        # self.step = step
        self.nh = len(self.sizes) - 2
        self.v_w = {}
        self.v_b = {}
        self.m_w = {}
        self.m_b = {}
        # Initializing
        for i in range(self.nh + 1):
            self.v_w[i + 1] = np.zeros((self.sizes[i + 1], self.sizes[i]))
            self.v_b[i + 1] = np.zeros((self.sizes[i + 1], 1))
            self.m_w[i + 1] = np.zeros((self.sizes[i + 1], self.sizes[i]))
            self.m_b[i + 1] = np.zeros((self.sizes[i + 1], 1))

    def learning_algoithms(self, x, y, dW, dB, update):

        if self.optimizer == "SGD":  # Stochastic gradient descent
            for i in range(self.nh + 1):
                self.W[i + 1] -= self.learning_rate * (dW[i + 1])
                self.B[i + 1] -= self.learning_rate * (dB[i + 1])

        if self.optimizer == "MBGD":  # Mini-batch gradient descent
            for i in range(self.nh + 1):
                self.W[i + 1] -= self.learning_rate * (dW[i + 1] / self.batch_size)
                self.B[i + 1] -= self.learning_rate * (dB[i + 1] / self.batch_size)

        if self.optimizer == "MGD":  # Momentum-based gradient descent
            beta = 0.9
            for i in range(self.nh + 1):
                # Updating history term
                self.v_w[i + 1] = beta * self.v_w[i + 1] + self.learning_rate * (
                    dW[i + 1] / self.batch_size
                )
                self.v_b[i + 1] = beta * self.v_b[i + 1] + self.learning_rate * (
                    dB[i + 1] / self.batch_size
                )
                # Updating weights and biases
                self.W[i + 1] -= self.v_w[i + 1]
                self.B[i + 1] -= self.v_b[i + 1]

        if self.optimizer == "NAG":  # Nestrov accelarated gradient descent
            beta = 0.9
            for i in range(self.nh + 1):
                # Computing look ahead term
                self.v_w[i + 1] = beta * self.v_w[i + 1]
                self.v_b[i + 1] = beta * self.v_b[i + 1]
                self.W[i + 1] = self.W[i + 1] - self.v_w[i + 1]
                self.B[i + 1] = self.B[i + 1] - self.v_b[i + 1]
            (dW, dB) = model.grad(
                x, y, self.activation_function, self.W, self.B, self.nh
            )
            # Updating weights and biases
            for i in range(self.nh + 1):
                # Updating history term
                self.v_w[i + 1] = beta * self.v_w[i + 1] + self.learning_rate * (
                    dW[i + 1] / self.batch_size
                )
                self.v_b[i + 1] = beta * self.v_b[i + 1] + self.learning_rate * (
                    dB[i + 1] / self.batch_size
                )
                # Updating weights and biases
                self.W[i + 1] -= self.v_w[i + 1]
                self.B[i + 1] -= self.v_b[i + 1]

        if self.optimizer == "RMSPROP":  # Root mean squared propagation
            beta = 0.9
            eps = 1e-8
            for i in range(self.nh + 1):
                # Updating history term
                self.v_w[i + 1] = beta * self.v_w[i + 1] + (1 - beta) * (
                    (dW[i + 1] / self.batch_size) ** 2
                )
                self.v_b[i + 1] = beta * self.v_b[i + 1] + (1 - beta) * (
                    (dB[i + 1] / self.batch_size) ** 2
                )
                # Updating weights and biases
                self.W[i + 1] -= (
                    self.learning_rate / np.sqrt(self.v_w[i + 1] + eps)
                ) * (dW[i + 1] / self.batch_size)
                self.B[i + 1] -= (
                    self.learning_rate / np.sqrt(self.v_b[i + 1] + eps)
                ) * (dB[i + 1] / self.batch_size)

        if self.optimizer == "ADAM":  # Adaptive moment estimation
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            for i in range(self.nh + 1):

                # Updating history term
                self.m_w[i + 1] = beta1 * self.m_w[i + 1] + (1 - beta1) * (
                    dW[i + 1] / self.batch_size
                )
                self.m_b[i + 1] = beta1 * self.m_b[i + 1] + (1 - beta1) * (
                    dB[i + 1] / self.batch_size
                )

                m_w_hat = self.m_w[i + 1] / (1 - np.power(beta1, update))
                m_b_hat = self.m_b[i + 1] / (1 - np.power(beta1, update))

                self.v_w[i + 1] = beta2 * self.v_w[i + 1] + (1 - beta2) * (
                    (dW[i + 1] / self.batch_size) ** 2
                )
                self.v_b[i + 1] = beta2 * self.v_b[i + 1] + (1 - beta2) * (
                    (dB[i + 1] / self.batch_size) ** 2
                )

                v_w_hat = self.v_w[i + 1] / (1 - np.power(beta2, update))
                v_b_hat = self.v_b[i + 1] / (1 - np.power(beta2, update))

                # Updating weights and biases
                self.W[i + 1] -= (
                    self.learning_rate / (np.sqrt(v_w_hat) + eps)
                ) * m_w_hat
                self.B[i + 1] -= (
                    self.learning_rate / (np.sqrt(v_b_hat) + eps)
                ) * m_b_hat

        if self.optimizer == "NADAM":  # Nestrov Adaptive moment estimation
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            for i in range(self.nh + 1):

                # Updating history term
                self.m_w[i + 1] = beta1 * self.m_w[i + 1] + (1 - beta1) * (
                    dW[i + 1] / self.batch_size
                )
                self.m_b[i + 1] = beta1 * self.m_b[i + 1] + (1 - beta1) * (
                    dB[i + 1] / self.batch_size
                )

                m_w_hat = self.m_w[i + 1] / (1 - np.power(beta1, update))
                m_b_hat = self.m_b[i + 1] / (1 - np.power(beta1, update))

                self.v_w[i + 1] = beta2 * self.v_w[i + 1] + (1 - beta2) * (
                    (dW[i + 1] / self.batch_size) ** 2
                )
                self.v_b[i + 1] = beta2 * self.v_b[i + 1] + (1 - beta2) * (
                    (dB[i + 1] / self.batch_size) ** 2
                )

                v_w_hat = self.v_w[i + 1] / (1 - np.power(beta2, update))
                v_b_hat = self.v_b[i + 1] / (1 - np.power(beta2, update))

                # Updating weights and biases
                self.W[i + 1] -= (self.learning_rate / (np.sqrt(v_w_hat) + eps)) * (
                    m_w_hat * beta1
                    + (((1 - beta1) * dW[i + 1]) / (1 - np.power(beta1, update)))
                )
                self.B[i + 1] -= (self.learning_rate / (np.sqrt(v_b_hat) + eps)) * (
                    m_b_hat * beta1
                    + (((1 - beta1) * dB[i + 1]) / (1 - np.power(beta1, update)))
                )

        return (self.W, self.B)
