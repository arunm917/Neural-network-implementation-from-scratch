import numpy as np
import pandas as pd
import activation_functions

sigmoid = activation_functions.sigmoid()

class gradients():
    
    def grad_w (self, W, b, x, y):
        y_pred = sigmoid(W, b, x)
        return (y_pred - y)*y_pred*(1-y_pred)*x
    
    def grad_b (self, W, b, x, y):
        y_pred = sigmoid(W, b, x)
        return (y_pred - y)*y_pred*(1-y_pred)
    