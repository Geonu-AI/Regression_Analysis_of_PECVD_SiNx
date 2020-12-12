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

GK2020_Ver1_CalledBack = keras.models.load_model('results/GK2020_Ver1.h5')

# Prediction with new inputs for future experiments
# using normalization parameters from training set


#Prediction with training set
ynew = GK2020_Ver1_CalledBack.predict(xscale)
y_pred = scaler_y.inverse_transform(ynew)
for i in range(0,len(y_pred)):
    print(y_pred[i,0])

