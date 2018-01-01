import tensorflow as tf
import os
import matplotlib.pyplot as plt

x1 = tf.placeholder(tf.float32, name = "x1")
x2 = tf.placeholder(tf.float32, name = "x2")
y = tf.placeholder(tf.float32, name = "y")

W = tf.Variable([0.1, 0.1], dtype=tf.float32)
b = tf.Variable([0.1], dtype=tf.float32)

x1_train = [0, 0, 1, 1]
x2_train = [0, 1, 0, 1]
y_train  = [0, 0, 0, 1]

y_hat = tf.sigmoid(W[0]*x1 + W[1]*x2+b)
loss = tf.reduce_sum(tf.squared_difference(y_hat, y))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(loss, {x1: x1_train, x2: x2_train, y:y_train}))

learn_rate = .5
optimizer = tf.train.GradientDescentOptimizer(learn_rate)
train = optimizer.minimize(loss)

log_path = os.path.dirname(os.path.abspath(__file__))
graph_writer = tf.summary.FileWriter(log_path , sess.graph)
n_epochs = 5000
cost_vector = []
for i in range(n_epochs):
    _, loss_value = sess.run([train, loss], {x1: x1_train, x2: x2_train, y:y_train})
    cost_vector.append(loss_value)

plt.plot(range(n_epochs), cost_vector)
plt.ylabel("cost")
plt.show()

W_opt, b_opt, loss_opt, y_hat= sess.run([W, b, loss, y_hat], {x1: x1_train, x2: x2_train, y:y_train})
print("W = {} , b = {} , loss= {}, y_hat= {}".format(W_opt, b_opt, loss_opt, y_hat))