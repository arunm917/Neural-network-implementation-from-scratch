import numpy as np
import DataPreprocessing
import ModelFinal

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
