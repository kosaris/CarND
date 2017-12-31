import tensorflow as tf

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

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

n_epochs = 5000
for i in range(n_epochs):
    sess.run(train, {x1: x1_train, x2: x2_train, y:y_train})
    #print(sess.run(loss))
W_opt, b_opt, loss_opt, y_hat= sess.run([W, b, loss, y_hat], {x1: x1_train, x2: x2_train, y:y_train})
print("W = {} , b = {} , loss= {}, y_hat= {}".format(W_opt, b_opt, loss_opt, y_hat))