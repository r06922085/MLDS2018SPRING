
# coding: utf-8

# In[1]:


# import library
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA


# In[2]:


def model_function(config):
    # combine the network
    # set placehold to put data and label
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
    # add hidden layer 1
    layer1 = tf.layers.dense(xs, 10, tf.nn.relu)
    # add hidden layer 2
    layer2 = tf.layers.dense(layer1, 15, tf.nn.relu)
    # add hidden layer 3
    layer3 = tf.layers.dense(layer2, 20, tf.nn.relu)
    # add hidden layer 4
    layer4 = tf.layers.dense(layer3, 15, tf.nn.relu)
    # add hidden layer 5
    layer5 = tf.layers.dense(layer4, 10, tf.nn.relu)
    # add output layer
    prediction = tf.layers.dense(layer5, 1, None)
    # define loss function and set optimizer
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    optimizer = tf.train.AdamOptimizer(0.001)
    train_step = optimizer.minimize(loss)
    # gradient of ne
    gradient_all = optimizer.compute_gradients(loss)  
    # compute gradient norm of every variable
    gradient = 0
    for (g, v) in gradient_all:
        if g is not None:
            gradient += tf.sqrt(tf.reduce_sum(tf.square(g)))

    # initialize variable
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    
    # create data and label
    # y = sin(5*math.pi*x) + x
    x_data = np.linspace(1,0,50000,endpoint=False)[:, np.newaxis]
    y_data = np.sin(5*math.pi*x_data)/(5*math.pi*x_data)
    
    # start to train
    loss_record = []
    gradient_record = []
    for step in range(700):
        _, loss_value, gradient_value = sess.run([train_step, loss, gradient], feed_dict={xs: x_data, ys: y_data})
        loss_record.append(loss_value)
        gradient_record.append(gradient_value)
        if step % 50 == 0:
            print("Iteration: %d Loss: %.8f Gradient norm: %.8f"%(step,loss_value,gradient_value))
    return loss_record, gradient_record


# In[3]:


def model_mnist(config):
    # load training and eval data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # define input
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    # layer 1
    layer1 = tf.layers.dense(inputs=xs,units=8,activation=tf.nn.relu)
    # layer 2
    layer2 = tf.layers.dense(inputs=layer1,units=200,activation=tf.nn.relu)
    # logits layer
    logits = tf.layers.dense(inputs=layer2, units=10)
    # output layer
    predictions = tf.nn.softmax(logits, name="softmax_tensor")

    # define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=logits))
    optimizer = tf.train.AdamOptimizer(0.001)
    train_step = optimizer.minimize(loss)
    # gradient of network (with NoneType)
    gradient_all = optimizer.compute_gradients(loss)  
    # compute gradient norm of every variable
    gradient = 0
    for (g, v) in gradient_all:
        if g is not None:
            gradient += tf.sqrt(tf.reduce_sum(tf.square(g)))

    # define accuracy
    correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    # start to train
    sess = tf.Session(config=config)
    # initialize variable
    init = tf.global_variables_initializer()
    sess.run(init)
    all_vars = tf.trainable_variables()
    
    loss_record = []
    gradient_record = []
    # run 100 epoch
    for step in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(1000)
        _, loss_value, gradient_value = sess.run([train_step, loss, gradient], feed_dict={xs:batch_xs, ys:batch_ys})
        loss_record.append(loss_value)
        gradient_record.append(gradient_value)
        if step%550 == 0:
            print("Iteration: %d Loss: %.8f Gradient norm: %.8f"%(step,loss_value,gradient_value))
    return loss_record, gradient_record
                
                


# In[4]:


def plotPart(gradient, loss, fileName):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.plot(gradient)
    ax1.set_ylabel('grad')
    ax2.plot(loss)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('loss')
    plt.savefig('report_data/'+fileName)
    plt.show()


# In[5]:


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    loss_function, gradient_function = model_function(config)
    loss_mnist, gradient_mnist = model_mnist(config)


# In[6]:


plotPart(gradient_function, loss_function, "function_gradient.png")
plotPart(gradient_mnist, loss_mnist, "mnist_gradient.png")

