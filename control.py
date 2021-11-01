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
from getPrediction import make_forecasts,inverse_transform, forecast_lstm , prepare_data , predict
from getDataFromLass import getLassData
import time

import paho.mqtt.client as mqtt
client = mqtt.Client()
client.connect("140.122.184.223", 8883, 60)
while(1):
        
    prediction = predict()
    print(prediction)
    thi = prediction[0][9][5]
    co2 = prediction[0][9][0]
    if( thi > 26 ) :
        client.publish("ac","on")
    if( thi < 22 ) :
        client.publish("ac","on")
    if( co2 > 1000) :
        client.publish("plan/fan","on")
    if( co2 < 500) :
        client.publish("plan/fan","off")
    time.sleep(60*30)


