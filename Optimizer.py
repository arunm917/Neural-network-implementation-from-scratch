import numpy as np

class Optimizer:
  def __init__(self,W, B, sizes, batch_size, learning_rate, optimizer):
    self.W = W
    self.B = B
    # self.dW = dW
    # self.dB = dB
    self.sizes = sizes
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.optimizer = optimizer
    # self.epoch = epoch
    # self.step = step
    self.nh = len(self.sizes) - 2
    self.v_w = {}
    self.v_b = {}
    self.m_w = {}
    self.m_b = {}
    #Initializing
    for i in range(self.nh + 1):
      self.v_w[i+1] = np.zeros((self.sizes[i+1], self.sizes[i]))
      self.v_b[i+1] = np.zeros((self.sizes[i+1],1))
      self.m_w[i+1] = np.zeros((self.sizes[i+1], self.sizes[i]))
      self.m_b[i+1] = np.zeros((self.sizes[i+1],1))

  def learning_algoithms(self,x, y, dW, dB, update):

    if self.optimizer == 'SGD':         # Stochastic gradient descent
      for i in range(self.nh + 1):
        self.W[i+1] -= self.learning_rate * (dW[i+1])
        self.B[i+1] -= self.learning_rate * (dB[i+1])

    if self.optimizer == 'MBGD':        # Mini-batch gradient descent
      for i in range(self.nh + 1):
        self.W[i+1] -= self.learning_rate * (dW[i+1]/self.batch_size)
        self.B[i+1] -= self.learning_rate * (dB[i+1]/self.batch_size)

    if self.optimizer == 'MGD':         # Momentum-based gradient descent
      beta = 0.9
      for i in range(self.nh + 1):
        # Updating history term
        self.v_w[i+1] = beta*self.v_w[i+1] + self.learning_rate * (dW[i+1]/self.batch_size)
        self.v_b[i+1] = beta*self.v_b[i+1] + self.learning_rate * (dB[i+1]/self.batch_size)
        # Updating weights and biases
        self.W[i+1] -= self.v_w[i+1]
        self.B[i+1] -= self.v_b[i+1]

    if self.optimizer == 'NAG':          # Nestrov accelarated gradient descent
      beta = 0.9
      for i in range(self.nh + 1):
        # Computing look ahead term
        self.v_w[i+1] = beta*self.v_w[i+1]
        self.v_b[i+1] = beta*self.v_b[i+1]
        self.W[i+1] = self.W[i+1] - self.v_w[i+1]
        self.B[i+1] = self.B[i+1] - self.v_b[i+1]
      (dW, dB) = model.grad(x, y, self.W, self.B)
        # Updating weights and biases
      for i in range(self.nh + 1):
        # Updating history term
        self.v_w[i+1] = beta*self.v_w[i+1] + self.learning_rate * (dW[i+1]/self.batch_size)
        self.v_b[i+1] = beta*self.v_b[i+1] + self.learning_rate * (dB[i+1]/self.batch_size)
        # Updating weights and biases
        self.W[i+1] -= self.v_w[i+1]
        self.B[i+1] -= self.v_b[i+1]

    if self.optimizer == 'RMSPROP':       # Root mean squared propagation
      beta = 0.9
      eps = 1e-8
      for i in range(self.nh + 1):
        # Updating history term
        self.v_w[i+1] = beta*self.v_w[i+1] + (1-beta) * ((dW[i+1]/self.batch_size)**2)
        self.v_b[i+1] = beta*self.v_b[i+1] + (1-beta) * ((dB[i+1]/self.batch_size)**2)
        # Updating weights and biases
        self.W[i+1] -= (self.learning_rate/np.sqrt(self.v_w[i+1] + eps)) * (dW[i+1]/self.batch_size)
        self.B[i+1] -= (self.learning_rate/np.sqrt(self.v_b[i+1] + eps)) * (dB[i+1]/self.batch_size)

    if self.optimizer == 'ADAM':           # Adaptive moment estimation
      beta1 = 0.9
      beta2 = 0.999
      eps = 1e-8
      for i in range(self.nh + 1):

        # Updating history term
        self.m_w[i+1] = beta1*self.m_w[i+1] + (1-beta1) * (dW[i+1]/self.batch_size)
        self.m_b[i+1] = beta1*self.m_b[i+1] + (1-beta1) * (dB[i+1]/self.batch_size)

        m_w_hat = self.m_w[i+1]/(1 - np.power(beta1,update))
        m_b_hat = self.m_b[i+1]/(1 - np.power(beta1,update))

        self.v_w[i+1] = beta2*self.v_w[i+1] + (1-beta2) * ((dW[i+1]/self.batch_size)**2)
        self.v_b[i+1] = beta2*self.v_b[i+1] + (1-beta2) * ((dB[i+1]/self.batch_size)**2)

        v_w_hat = self.v_w[i+1]/(1 - np.power(beta2,update))
        v_b_hat = self.v_b[i+1]/(1 - np.power(beta2,update))

        # Updating weights and biases
        self.W[i+1] -= (self.learning_rate/(np.sqrt(self.v_w[i+1]) + eps)) * m_w_hat
        self.B[i+1] -= (self.learning_rate/(np.sqrt(self.v_b[i+1]) + eps)) * m_b_hat

    if self.optimizer == 'NADAM':          # Nestrov Adaptive moment estimation
      beta1 = 0.9
      beta2 = 0.999
      eps = 1e-8
      for i in range(self.nh + 1):

        # Updating history term
        self.m_w[i+1] = beta1*self.m_w[i+1] + (1-beta1) * (dW[i+1]/self.batch_size)
        self.m_b[i+1] = beta1*self.m_b[i+1] + (1-beta1) * (dB[i+1]/self.batch_size)

        m_w_hat = self.m_w[i+1]/(1 - np.power(beta1,update))
        m_b_hat = self.m_b[i+1]/(1 - np.power(beta1,update))

        self.v_w[i+1] = beta2*self.v_w[i+1] + (1-beta2) * ((dW[i+1]/self.batch_size)**2)
        self.v_b[i+1] = beta2*self.v_b[i+1] + (1-beta2) * ((dB[i+1]/self.batch_size)**2)

        v_w_hat = self.v_w[i+1]/(1 - np.power(beta2,update))
        v_b_hat = self.v_b[i+1]/(1 - np.power(beta2,update))

        # Updating weights and biases
        self.W[i+1] -= (self.learning_rate/(np.sqrt(self.v_w[i+1]) + eps)) * (m_w_hat*beta1 + (((1-beta1)*dW[i+1])/(1-np.power(beta1,update))))
        self.B[i+1] -= (self.learning_rate/(np.sqrt(self.v_b[i+1]) + eps)) * (m_b_hat*beta1 + (((1-beta1)*dB[i+1])/(1-np.power(beta1,update))))
    
    return(self.W, self.B)