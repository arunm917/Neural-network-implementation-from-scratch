import numpy as np
import pandas as pd

class sigmoid:
    def sigmoid_compute(self, W, b, x):
        return 1/1+np.exp(-(np.dot(W,x) + b))