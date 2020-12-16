# NN_Model with Keras

import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# from IPython.display import SVG


def GK2020(input_shape):
    """
    Implementation of the GK2020.
    Arguments:
    input_shape = shape of the dataset (# of input X)
    Returns:
    a Model() instance in Keras
    """
    X_input = Input(input_shape)
    X = Dense(2, activation='tanh',kernel_initializer='normal', name='dense_1')(X_input)
    X = Dense(1, activation='linear', name='dense_output')(X)
    model = Model(inputs = X_input, outputs = X, name = 'GK2020')

    return model

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

# Hyper Parameter selection

GK2020_Ver1 = GK2020(x.shape[1:])
GK2020_Ver1.compile(loss='mse', optimizer = 'adam', metrics=['accuracy'])


for i in range(1,5):
    GK2020_Ver1.fit(X_train, Y_train, epochs= 30*i, batch_size= 50, verbose= 0)
    y_pred = []
    ynew = GK2020_Ver1.predict(xscale)
    y_pred_np = scaler_y.inverse_transform(ynew)
    for j in range(0, len(y_pred_np)):
        y_pred.append(y_pred_np[j, 0])
    plt.plot(y_pred_np, linestyle='--', linewidth = 0.8, label='Y_pred, a='+str(30*i))

y_exp = []
for i in range(0, len(yscale)):
    y_exp.append(y[i, 0])
plt.plot(y_exp, color='blue', label='Y_exp')

# y_pred = []
# ynew = GK2020_Ver1.predict(xscale)
# y_pred_np = scaler_y.inverse_transform(ynew)
# for i in range(0,len(y_pred_np)):
#     y_pred.append(y_pred_np[i,0])
# plt.plot(y_pred_np, color = 'red', linewidth = 0.7, label='Y_pred_last')

plt.legend()
plt.xlabel('PECVD Deposition Exp.')
plt.ylabel('Tensile Stress (MPa)')
plt.title('Neural Network Prediction vs. Experimental Results')
plt.show()



## Model Evaluation
# preds = GK2020_Ver1.evaluate(x = xscale, y= yscale)
# print()
# print("Loss=", str(preds[0]))
# print("Test Accuracy=", str(preds[1]))

## Model Information
# GK2020_Ver1.summary()
# plot_model(GK2020_Ver1, to_file='GK2020_Ver1.png')
# SVG(model_to_dot(GK2020_Ver1).create(prog='dot', format='svg'))


# # Save the model!
# GK2020_Ver1.save_weights('results/Models_Trained/GK2020_Ver1_weights (20000 epochs, 2tanh + 1linear).h5')
# GK2020_Ver1.save('results/Models_Trained/GK2020_Ver1 (20000 epochs, 2tanh + 1linear).h5')
#



