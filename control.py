import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras import optimizers
from math import sqrt
from matplotlib import pyplot
from numpy import array
import numpy as np
import pdb
from get_data import make_forecasts,inverse_transform, forecast_lstm , prepare_data


# configure
n_lag = 10 #n_in/n_times
n_seq = 10 #n_out
n_test = 10
n_epochs = 100
n_batch = 1
n_neurons = 1
n_features = 8

n_obs = n_lag * n_features
# load model and import data
model =  load_model('modelall.h5')

series = pd.read_csv('AirData2020-12-11(1).csv', header=0, index_col=0)

# prepare data
scaler, supervised_values = prepare_data(series, n_test, n_lag, n_seq, n_features)

# make forecasts
forecasts = make_forecasts(model, n_batch, supervised_values, n_lag, n_seq, n_features)

# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+(n_seq-1), n_seq, n_features)
forecasts = np.array(forecasts)
print(forecasts)


for i in range(n_seq):
    predicted = [forecast[i,0] for forecast in forecasts]
    print(predicted)

