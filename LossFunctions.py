import numpy as np

class LossFunctions:
  def MSE(self, y, y_pred):
    return (np.sum((y-y_pred)**2))/len(y)
  
  def cross_entropy(self, y, y_pred):
    # clipping y_pred to prevent log(0) errors
    #print(y.shape,y_pred.shape)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    ce_loss = np.mean(np.sum(np.multiply(-y,np.log(y_pred)), axis = 0))
    #print(ce_loss)
    return(ce_loss)
loss_functions = LossFunctions()