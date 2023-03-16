import numpy as np
import pandas as pd
import gdown
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm.notebook import trange, tqdm
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from sklearn.metrics import accuracy_score, log_loss
import wandb
import random

