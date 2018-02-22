from process_eeg import get_features_and_labels
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import matplotlib.animation as animation

train_x,test_x,train_y,test_y = get_features_and_labels(0.2)

n_nodes_hl1 = 100
n_nodes_hl2 = 100

n_classes = 2
batch_size = 500
hm_epochs = 20

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.get_variable('W1',shape = [train_x.shape[1], n_nodes_hl1],
                                            initializer=tf.contrib.layers.xavier_initializer()),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.get_variable('W2', shape = [n_nodes_hl1, n_nodes_hl2],
                                            initializer = tf.contrib.layers.xavier_initializer()),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum': None,
                'weight': tf.get_variable('W3', shape = [n_nodes_hl2, n_classes],
                                          initializer = tf.contrib.layers.xavier_initializer()),
                'bias': tf.Variable(tf.random_normal([n_classes])), }

#plot figure
# fig, ax = plt.subplots(1,1)
# x_axis = []
# y_axis = []

# Nothing changes
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction, labels = y)) + \
           0.01*tf.nn.l2_loss(hidden_1_layer['weight']) + 0.01 * tf.nn.l2_loss(hidden_2_layer['weight']) + \
           0.01*tf.nn.l2_loss(output_layer['weight'])
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(tf.local_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0

            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x,
                                                                y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            # print('minibatch Accuracy:', accuracy.eval({x: test_x, y: test_y}))

            #plot scatter
            # x_axis.append(epoch)
            # y_axis.append(epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x)
# plt.plot(x_axis,y_axis)
# plt.show()