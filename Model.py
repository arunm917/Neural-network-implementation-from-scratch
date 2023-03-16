import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm.notebook import trange, tqdm
import ActivationFunctions as act
import LossFunctions as loss_functions
import Optimizer as optimizer
import ParameterInitialization as initial



class Model():
    
  def forward_prop(self, x, activation_function):
    self.A = {}
    self.H = {}
    self.x = x
    loss_batch = 0
    self.H[0] = self.x.T

    for i in range(self.nh + 1):
      self.A[i+1] = np.matmul(self.W[i+1], self.H[i]) + self.B[i+1]
      self.H[i+1] = act.activation(self.A[i+1], activation_function)

    self.y_pred = act.softmax(self.A[i+1])
    #print(y_pred)
    #print('Y = ', self.y.shape, 'Y_pred =', y_pred.shape)
    return(self.y_pred)
  
  def grad(self, x, y, activation_function, W, B):

    if W is None:
      W = self.W
    if B is None:
      B = self.B

    y_pred = self.forward_prop(x, activation_function)
    self.dW = {}
    self.dB = {}
    self.dA = {}
    self.dH = {}
    self.dA[self.nh + 1] = y_pred - y.T

    for i in range(self.nh + 1 , 0, -1):
      self.dW[i]      = np.matmul(self.dA[i],self.H[i-1].T)
      self.dB[i]      = np.sum(self.dA[i], axis = 1).reshape(-1,1)
      self.dH[i-1]    = np.matmul(W[i].T,self.dA[i])
      self.dA[i-1]    = np.multiply(self.dH[i-1], act.der_activation(self.H[i-1], activation_function))
    return (self.dW, self.dB)
  
  def fit(self, X, Y, X_val, Y_val, epochs, learning_rate, hidden_layers, neurons_per_layer, batch_size, optimizer, initialization, activation_function):
    
    # x = X[0:batch_size,:]
    # y = Y[0:batch_size,:]
    #self.X = X
    self.loss_epoc_store = []
    self.nx = X.shape[1]
    self.ny = Y.shape[1]
    self.nh = hidden_layers
    self.neurons = neurons_per_layer
    hidden_layer_sizes = [self.neurons]*self.nh
    self.sizes = [self.nx] + hidden_layer_sizes + [self.ny]
    print(self.sizes)
    if optimizer == 'SGD':
      batch_size =1

    step_size = len(X)/batch_size

    self.W, self.B = initial.initialize(self.sizes, initialization)
    opt = Optimizer(self.W, self.B, self.sizes, batch_size, learning_rate, optimizer)

    for i in trange(epochs, total=epochs, unit="epoch"):
      epoch = i+1
      step = 0
      # print('epoch = ', epoch)
      loss_epoc = 0
      self.loss_batch_store = []

      for i in range(0,len(X),batch_size):
        step += 1
        # print('step = ', step)
        self.x = X[i:i+batch_size,:]
        self.y = Y[i:i+batch_size,:]
        #print('y =', self.y.shape)
        (self.dW, self.dB) = self.grad(self.x, self.y, activation_function, W = None, B = None)
        self.W, self.B = opt.learning_algoithms(self.x, self.y, self.dW, self.dB, step)
        # Predicting loss for each batch
        loss_batch = loss_functions.cross_entropy(self.y.T ,self.y_pred)
        self.loss_batch_store.append(loss_batch)
  
      loss_epoc = np.sum(self.loss_batch_store)/step_size
      print('training loss = ', round(loss_epoc,4))
      self.loss_epoc_store.append(loss_epoc)

      # Predicting validation loss
      y_pred_val = self.forward_prop(X_val, activation_function)
      Y_pred_val = np.array(y_pred_val.T).squeeze()
      Y_pred_val = np.argmax(Y_pred_val,1)
      #print(Y_pred_val[0:10], Y_val[0:10])
      accuracy_val = accuracy_score(Y_pred_val,Y_val)
      print('validation accuracy = ', round(accuracy_val,2))
      return(accuracy_val, loss_epoc)
    
    plt.plot(self.loss_epoc_store)
    plt.xlabel('Epochs')
    plt.ylabel('log_loss')
    plt.show()

model = Model()