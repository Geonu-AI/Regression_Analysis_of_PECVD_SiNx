from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
import numpy as np


database = np.genfromtxt('database/Data Set.csv',delimiter=',')
x = database[1:,2:7] # Input (m, # of inputs)
y = database[1:,7]  # Output (m, # of outputs)

# Input data scaling (Normalization)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
y = np.reshape(y, (-1,1))
print(scaler_x.fit(x))
xscale=scaler_x.transform(x)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)
X_train, X_test, Y_train, Y_test = train_test_split(xscale, yscale)

GK2020_Ver1_CalledBack = keras.models.load_model('results/Models_Trained/GK2020_Ver1.h5')

# Prediction with new inputs for future experiments
# using normalization parameters from training set

# # Prediction again with training set used already
# ynew = GK2020_Ver1_CalledBack.predict(xscale)
# y_pred = scaler_y.inverse_transform(ynew)
# for i in range(0,len(y_pred)):
#     print(y_pred[i,0])

## Prediction with new data set
# Method 1
TS_X_std  = xscale.std(axis=0)
TS_X_mean = xscale.mean(axis=0)
TS_X_max  = scaler_x.data_max_
TS_X_min  = scaler_x.data_min_
TS_Y_std  = yscale.std(axis=0)
TS_Y_mean = yscale.mean(axis=0)
TS_Y_max  = scaler_y.data_max_
TS_Y_min  = scaler_y.data_min_

x_new_test = database[1:,2:7] ## new data set we want to test
x_new_scaled = (x_new_test - TS_X_min) / (TS_X_max - TS_X_min) * 1 + 0

y_new_scaled = GK2020_Ver1_CalledBack(x_new_scaled)
y_new = y_new_scaled * (TS_Y_max - TS_Y_min) + TS_Y_min ## prediction using model trained

for i in range(0,len(y_new)):
    print(y_new.numpy()[i,0])

# # Method 2 (I guess scaler_x, and scaler_y contain mean & std of trainingset and able to using trainingset mean & std for new data scaling) """
#
# x_new_test = database[1:,2:7] # New data set we want to test
# x_new_scaled = scaler_x.transform(x_new_test)
# y_new_scaled = GK2020_Ver1_CalledBack(x_new_scaled)
# y_new = scaler_y.inverse_transform(y_new_scaled) # prediction using the model trained
#
# for i in range(0,len(y_new)):
#     print(y_new[i,0])
