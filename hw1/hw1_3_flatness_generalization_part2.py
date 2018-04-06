
# coding: utf-8

# In[1]:


# import library
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA


# In[14]:


def model(config, mode, batch_size, var_model1=None, var_model2=None):
    # reset graph and create a session
    tf.reset_default_graph()
    sess = tf.Session(config=config)
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

    # define accuracy
    correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # gradient of loss to input
    input_grad = tf.gradients(loss, xs)   
    # compute frobenius norm of gradients of loss to input
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(input_grad[0])))

    # start to train
    # initialize variable
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # run 100 epoch
    epoch_num = int((55000/batch_size)*100)
    for step in range(epoch_num):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, loss_value = sess.run([train_step, loss], feed_dict={xs:batch_xs, ys:batch_ys})
        if step%(int(55000/batch_size)) == 0:
            print("Epoch: %d Loss: %.8f"%(step/(55000/batch_size),loss_value))

    # start to test
    # load training data and testing data
    train_xs = mnist.train.images
    train_ys = mnist.train.labels
    test_xs = mnist.test.images
    test_ys = mnist.test.labels
        
    # run model and compute accuracy
    train_loss_value, train_accuracy_value = sess.run([loss,accuracy], feed_dict={xs:train_xs, ys:train_ys})
    test_loss_value, test_accuracy_value = sess.run([loss,accuracy], feed_dict={xs:test_xs, ys:test_ys})
    print("Train loss: %.8f Train accuracy: %.8f\nTest loss: %.8f Test accuracy: %.8f"%(train_loss_value,train_accuracy_value,test_loss_value,test_accuracy_value))
    # run model and compute sensitivity
    sensitivity = sess.run(gradient_norm, feed_dict={xs:test_xs, ys:test_ys})
    print("Sensitivity: %.8f"%sensitivity)
    return train_loss_value, train_accuracy_value, test_loss_value, test_accuracy_value, sensitivity


# In[16]:


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    train_loss_record = []
    train_accuracy_record = []
    test_loss_record = []
    test_accuracy_record = []
    sensitivity_record = []
    for i in range(5):
        train_loss, train_accuracy, test_loss, test_accuracy, sensitivity= model(config,'train',(5*(10**i)))
        train_loss_record.append(train_loss)
        train_accuracy_record.append(train_accuracy)
        test_loss_record.append(test_loss)
        test_accuracy_record.append(test_accuracy)
        sensitivity_record.append(sensitivity)


# In[20]:


def plotFunction(sensitivity, train_loss, train_accuracy, test_loss, test_accuracy, fileName):
    index = [5,50,500,5000,50000]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(index, train_loss,'b')
    ax1.plot(index, test_loss,'b--')
    ax1.set_xscale("log")
    ax1.set_xlabel('batch size')
    ax1.set_ylabel('loss')
    
    # use the same x axis
    ax2 = ax1.twinx()
    ax2.plot(index, sensitivity, 'r')
    ax2.set_xscale("log")
    ax2.set_ylabel('sensitivity')
    ax2.set_xlabel('batch size')
    plt.savefig('report_data/'+fileName+'_loss.png')
    plt.show()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(index, train_accuracy,'b')
    ax1.plot(index, test_accuracy,'b--')
    ax1.set_xlabel('batch size')
    ax1.set_ylabel('accuracy')
    ax1.set_xscale("log")
    
    # use the same x axis
    ax2 = ax1.twinx()
    ax2.plot(index, sensitivity, 'r')
    ax2.set_ylabel('sensitivity')
    
    ax2.set_xscale("log")
    plt.savefig('report_data/'+fileName+'_accuracy.png')
    plt.show()


# In[21]:


plotFunction(sensitivity_record, train_loss_record, train_accuracy_record, test_loss_record, test_accuracy_record, 'hw1_3_3_part2')

