# CS6910_Assignment1
In this project an artificial neural network model has been implemeted from scratch using numpy. The model can take in different hyperparameters and trains the neural network to a traing dataset and the learnt parameters can be used to make predictions on unseen data. In this implementation the model has been trained and tested on the fashion-mnist dataset.

1. train.py file is the main file to run the program
2. It contains options to enter hyperparameters and call the fit function on the model to train the model
3. train.py also contains the implementation of wandb sweeps used for hyperparameter tuning. The choices for different hyperparameters to be turned have been mentioned in this file. These can be modified for different datasets.
4. ModelFinal.py contains the implementation of the forward and back propagation algorithms and the learning algorithms
5. DataPreprocessing.py contains the implementation of downloading data from different datasets
6. LossFunctions.py contain the different loss functions explored
7. ParameterInitializations.py contains the different intitialization algorithms implemented
8. ActivationFunctions.py contains all the different activations functions implemented
