
# coding: utf-8

# In[ ]:


# import library
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random 
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA

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
    test_loss_record = []
    gradient_record = []
    
    #shuffle the labels
    for i in mnist.train.labels:
        prop = 1
        if (random.randint(1,10) <= (prop*10)):
            for index in range(0,10):
                i[index] = 0
            i[random.randint(0,9)] = 1
 
    #run 100 epoch
    for step in range(5500):
        batch_xs, batch_ys = mnist.train.next_batch(1000)
        _, loss_value, gradient_value = sess.run([train_step, loss, gradient], feed_dict={xs:batch_xs, ys:batch_ys})
        test_loss_value = sess.run([loss], feed_dict={xs:mnist.test.images, ys:mnist.test.labels})
        
        loss_record.append(loss_value)
        test_loss_record.append(test_loss_value)
        gradient_record.append(gradient_value)
        if step%55 == 0:
            print("Iteration: %d Loss: %.8f Gradient norm: %.8f"%(step,loss_value,gradient_value))
    return loss_record, gradient_record,test_loss_record
    
def plotPart(gradient, loss,test_loss, fileName):
    fig = plt.figure()
    ax2 = fig.add_subplot(1,1,1)
    ax2.plot(loss)
    ax2.plot(test_loss)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('loss')
    plt.savefig('report_data/'+fileName)
    plt.show()
    
if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    loss_mnist, gradient_mnist,test_loss_mnist = model_mnist(config)
    plotPart(gradient_mnist,loss_mnist,test_loss_mnist,'hw1_1_3_mnist_loss')

