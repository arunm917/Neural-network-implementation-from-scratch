import numpy as np


class LossFunctions:
    def MSE(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def cross_entropy(self, y, y_pred):
        # clipping y_pred to prevent log(0) errors
        # print(y.shape,y_pred.shape)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        ce_loss = np.mean(np.multiply(-y, np.log(y_pred + 0.00001)))
        # print(ce_loss)
        return ce_loss
