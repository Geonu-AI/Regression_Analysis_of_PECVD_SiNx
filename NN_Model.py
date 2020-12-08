# NN_Model with Keras
## Reference: https://datascienceplus.com/keras-regression-based-neural-networks/

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


database = np.genfromtxt('database/Data Set.csv',delimiter=',')
x = database[1:,2:7] # Input
y = database[1:,7]  # Output
y = np.reshape(y, (-1,1))

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(x))
xscale=scaler_x.transform(x)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)


model = Sequential()
model.add(Dense(2, input_dim = 5, kernel_initializer='normal', activation='sigmoid', name='dense_1'))
model.add(Dense(1, activation='linear',name='dense_output'))

model.compile(loss='mse', optimizer = 'adam', metrics=['mse','mae'])
history = model.fit(X_train, y_train, epochs=1000, batch_size=50,  verbose=1, validation_split=0.2)

print(history.history.keys())

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



#Evaluation ...
Xnew = database[1:,2:7]
Xnew= scaler_x.transform(Xnew)
ynew= model.predict(Xnew)
#invert normalize
ynew = scaler_y.inverse_transform(ynew)
Xnew = scaler_x.inverse_transform(Xnew)

for i in range(0,43):
    print(ynew[i,0])
# print(ynew)
# for i in range(0,len(database[1:,7])):
#     print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
