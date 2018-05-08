
# coding: utf-8

# In[1]:


# import library
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data
from functools import reduce
from operator import mul


# In[2]:


def model1(config):
    # load training and eval data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # define input
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    # convolutional layer 1
    conv1 = tf.layers.conv2d(inputs=x_image,filters=3,kernel_size=[3,3],padding='same',activation=tf.nn.relu)
    # pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],strides=2)
    # dense layer
    pool1_flat = tf.reshape(pool1, [-1,14*14*3])
    # logits layer
    logits = tf.layers.dense(inputs=pool1_flat, units=10)
    # output layer
    predictions = tf.nn.softmax(logits, name="softmax_tensor")
    # define loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=logits))

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # define accuracy
    correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initialize variable
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # start to train
    loss_record = []
    accuracy_record = []
    # load validation set
    validate_feed = {xs: mnist.validation.images, ys: mnist.validation.labels}
    for step in range(55000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, loss_value = sess.run([train_step, loss], feed_dict={xs:batch_xs, ys:batch_ys})
        if step % 550 == 0:
            accuracy_value = sess.run(accuracy, feed_dict=validate_feed)
            loss_record.append(loss_value)
            accuracy_record.append(accuracy_value)
            print("Epoch: %d Loss: %f Accuracy: %f"%(step/550,loss_value,accuracy_value))

    return loss_record,accuracy_record


# In[3]:


def model2(config):
    # load training and eval data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # define input
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    # convolutional layer 1
    conv1 = tf.layers.conv2d(inputs=x_image,filters=16,kernel_size=[5,5],padding='same',activation=tf.nn.relu)
    # pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],strides=2)
    # convolutional layer 2
    conv2 = tf.layers.conv2d(inputs=pool1,filters=4,kernel_size=[5,5],padding='same',activation=tf.nn.relu)
    # pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2],strides=2)
    # dense layer
    pool1_flat = tf.reshape(pool2, [-1,7*7*4])
    dense = tf.layers.dense(inputs=pool1_flat, units=25)
    # logits layer
    logits = tf.layers.dense(inputs=dense, units=10)
    # output layer
    predictions = tf.nn.softmax(logits, name="softmax_tensor")
    

    # define loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=logits))

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # define accuracy
    correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initialize variable
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # start to train
    loss_record = []
    accuracy_record = []
    # load validation set
    validate_feed = {xs: mnist.validation.images, ys: mnist.validation.labels}
    for step in range(55000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, loss_value = sess.run([train_step, loss], feed_dict={xs:batch_xs, ys:batch_ys})
        if step % 550 == 0:
            accuracy_value = sess.run(accuracy, feed_dict=validate_feed)
            loss_record.append(loss_value)
            accuracy_record.append(accuracy_value)
            print("Epoch: %d Loss: %f Accuracy: %f"%(step/550,loss_value,accuracy_value))
    return loss_record,accuracy_record


# In[ ]:


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    loss1,accuracy1 = model1(config)
    loss2,accuracy2 = model2(config)

    # visualize the target function and result
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(loss1, 'r',label='shallow')
    ax1.plot(loss2, 'b',label='deep')
    ax1.legend(loc="upper right")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    plt.savefig("report_data/hw1_1_mnist_loss.png")
    plt.show()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot(accuracy1, 'r',label='shallow')
    ax2.plot(accuracy2, 'b',label='deep')
    ax2.legend(loc="lower right")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    plt.savefig("report_data/hw1_1_mnist_accuracy.png")
    plt.show()

