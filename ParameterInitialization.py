import numpy as np

class ParameterInitialization:
  
  def initialize(self, sizes, initialization_alg):

    self.W = {}
    self.B = {}
    self.sizes = sizes
    self.nh = len(self.sizes)-2

    if initialization_alg =='RANDOM':                       # Random initialization
      for i in range(self.nh + 1):
        self.W[i+1] = np.random.randn(self.sizes[i+1], self.sizes[i])
        self.B[i+1] = np.zeros((self.sizes[i+1],1))

      return(self.W, self.B)
  
    if initialization_alg == 'XAVIER':                      # Xavier initialization
      for i in range(self.nh + 1):
        std_dev_xavier = np.sqrt(2.0 / (self.sizes[i] + self.sizes[i+1]))
        self.W[i+1] = np.random.normal(loc=0, scale=std_dev_xavier, size=(self.sizes[i+1], self.sizes[i]))
        self.B[i+1] = np.zeros((self.sizes[i+1],1))

      return(self.W, self.B)

    if initialization_alg == 'HE':                          # He initialization
      print('HE')
      for i in range(self.nh + 1):
        std_dev_he = np.sqrt(2.0 / self.sizes[i])
        self.W[i+1] = np.random.randn(self.sizes[i+1], self.sizes[i]) * std_dev_he
        self.B[i+1] = np.zeros((self.sizes[i+1],1))

      return(self.W, self.B)