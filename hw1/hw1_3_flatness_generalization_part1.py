
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

    
    
    # all variables' name
    all_vars_name = ['dense/kernel:0','dense/bias:0','dense_1/kernel:0','dense_1/bias:0','dense_2/kernel:0','dense_2/bias:0']
    if mode == 'train':
        # start to train
        # initialize variable
        init = tf.global_variables_initializer()
        sess.run(init)
        all_vars_value = []
    
        # run 100 epoch
        epoch_num = int((55000/batch_size)*100)
        for step in range(epoch_num):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, loss_value = sess.run([train_step, loss], feed_dict={xs:batch_xs, ys:batch_ys})
            if step%550 == 0:
                print("Epoch: %d Loss: %.8f"%((step/550),loss_value))
        # get the final parameters of model
        for var_name in all_vars_name:
            var = [v for v in tf.global_variables() if v.name == var_name][0]
            all_vars_value.append(sess.run(var))

        return all_vars_value
    
    elif mode == 'test':
        # start to test
        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        # load training data and testing data
        train_xs = mnist.train.images
        train_ys = mnist.train.labels
        test_xs = mnist.test.images
        test_ys = mnist.test.labels
        
        # compute interpolation for each alpha
        for alpha in (np.linspace(-1,2,1000)):
            for i in range(len(all_vars_name)):
                # calculate new variable value by interpolation
                new_value = var_model1[i]*(1-alpha) + var_model2[i]*alpha
                var = [v for v in tf.global_variables() if v.name == all_vars_name[i]][0]
                # assign new value to variable
                sess.run(tf.assign(var, new_value)) 
            train_loss_value, train_accuracy_value = sess.run([loss,accuracy], feed_dict={xs:train_xs, ys:train_ys})
            test_loss_value, test_accuracy_value = sess.run([loss,accuracy], feed_dict={xs:test_xs, ys:test_ys})
            print("Train loss: %.8f Train accuracy: %.8f\nTest loss: %.8f Test accuracy: %.8f"%(train_loss_value,train_accuracy_value,test_loss_value,test_accuracy_value))
            train_loss.append(train_loss_value)
            train_accuracy.append(train_accuracy_value)
            test_loss.append(test_loss_value)
            test_accuracy.append(test_accuracy_value)
    return train_loss, train_accuracy, test_loss, test_accuracy


# In[6]:


def plotFunction(train_loss, train_accuracy, test_loss, test_accuracy, fileName):
    alpha = np.linspace(-1,2,1000)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(alpha, train_loss, 'b', label='train')
    ax1.plot(alpha, test_loss, 'b--', label='test')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('loss')
    
    # use the same x axis
    ax2 = ax1.twinx()
    ax2.plot(alpha, train_accuracy, 'r')
    ax2.plot(alpha, test_accuracy, 'r--')
    ax2.set_ylabel('accuracy')
    
    plt.savefig('report_data/'+fileName)
    plt.show()


# In[4]:


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    var_model1 = model(config,'train',100)
    var_model2 = model(config,'train',1000)
    train_loss, train_accuracy, test_loss, test_accuracy = model(config,'test',0,var_model1,var_model2)


# In[7]:


plotFunction(train_loss, train_accuracy, test_loss, test_accuracy, 'hw1_3_3_part1.png')

