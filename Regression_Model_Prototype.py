# Linear Regression with Tensorflow

import tensorflow as tf
import numpy as np
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

database = np.genfromtxt('database/Data Set.csv',delimiter=',')

x1 = database[1:,2]/600 # RF POWER
x2 = database[1:,3]/900 # Pressure
x3 = database[1:,4]/120 # SiH4 sccm
x4 = database[1:,5]/75 # NH3 sccm
x5 = database[1:,6]/1500 # N2 sccm
y = database[1:,7]/100  # Tensile Stress
n = len(y)

X1 = tf.compat.v1.placeholder(tf.float32)
X2 = tf.compat.v1.placeholder(tf.float32)
X3 = tf.compat.v1.placeholder(tf.float32)
X4 = tf.compat.v1.placeholder(tf.float32)
X5 = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)

W1 = tf.Variable(np.random.randn(), name = "W1")
W2 = tf.Variable(np.random.randn(), name = "W2")
W3 = tf.Variable(np.random.randn(), name = "W3")
W4 = tf.Variable(np.random.randn(), name = "W4")
W5 = tf.Variable(np.random.randn(), name = "W5")
b = tf.Variable(np.random.randn(), name = "b")

learning_rate = 0.05
training_epochs = 300

# Hypothesis
### y_pred = tf.add(tf.add(tf.add(tf.add(tf.add(tf.multiply(X1, W1), tf.multiply(X2, W2)), tf.multiply(X3, W3)), tf(multiply(X4,W4))),tf(multiply(X5,W5))),b)
y_pred = X1*W1 + X2*W2 + X3*W3 + X4*W4 + X5*W5 + b

# Mean Squared Error Cost Function
cost = tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * n)

# Gradient Descent Optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Global Variables Initializer
init = tf.compat.v1.global_variables_initializer()

# Starting the Tensorflow Session
with tf.compat.v1.Session() as sess:
    # Initializing the Variables
    sess.run(init)

    # Iterating through all the epochs
    for epoch in range(training_epochs):

        # Feeding each data point into the optimizer using Feed Dictionary
        for (_x1, _x2, _x3, _x4, _x5, _y) in zip(x1, x2, x3, x4, x5, y):
            sess.run(optimizer, feed_dict={X1:_x1, X2:_x2,X3:_x3,X4:_x4,X5:_x5, Y:_y})

            # Displaying the result after every 50 epochs
        if (epoch + 1) % 50 == 0:
            # Calculating the cost a every epoch
            c = sess.run(cost, feed_dict={X1:_x1, X2:_x2,X3:_x3,X4:_x4,X5:_x5, Y:_y})
            print("Epoch", (epoch + 1), ": cost =", c, "W1 =", sess.run(W1), "W2 =", sess.run(W2),"W3 =", sess.run(W3),"W4 =", sess.run(W4),"W5 =", sess.run(W5),"b =", sess.run(b))

            # Storing necessary values to be used outside the Session
    training_cost = sess.run(cost, feed_dict={X1:_x1, X2:_x2,X3:_x3,X4:_x4,X5:_x5, Y:_y})
    weight1 = sess.run(W1)
    weight2 = sess.run(W2)
    weight3 = sess.run(W3)
    weight4 = sess.run(W4)
    weight5 = sess.run(W5)
    bias = sess.run(b)

#Normalized Weights
print(weight1, weight2, weight3, weight4, weight5, bias)
