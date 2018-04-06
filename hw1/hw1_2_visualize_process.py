
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

    # define loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=logits))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # define accuracy
    correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    # start to train
    sess = tf.Session(config=config)
    
    # record 8 training event
    hidden_layer1 = []
    whole_model = []
    accuracy_record = []
    # load validation set
    validate_feed = {xs: mnist.validation.images, ys: mnist.validation.labels}
    for i in range(8):
        # initialize variable
        init = tf.global_variables_initializer()
        sess.run(init)
        all_vars = tf.trainable_variables()
        
        # run 100 epoch
        for step in range(55000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, loss_value = sess.run([train_step, loss], feed_dict={xs:batch_xs, ys:batch_ys})
            # record every 5 epoch
            if step % 2750 == 0:
                layer1_each_epoch = np.array([])
                whole_each_epoch = np.array([])
                accuracy_value = sess.run(accuracy, feed_dict=validate_feed)
                accuracy_record.append(accuracy_value)
                print("Epoch: %d Loss: %f Accuracy: %f"%((step/550),loss_value,accuracy_value))
                layer1_each_epoch = np.append(layer1_each_epoch,(sess.run('dense/kernel:0').flatten()))
                layer1_each_epoch = np.append(layer1_each_epoch,(sess.run('dense/bias:0').flatten()))
                for v in all_vars:
                    whole_each_epoch = np.append(whole_each_epoch,(sess.run(v).flatten()))
                hidden_layer1.append(layer1_each_epoch)
                whole_model.append(whole_each_epoch)
    return accuracy_record, hidden_layer1, whole_model
                
                


# In[3]:


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    accuracy_record,hidden_layer1, whole_model = model_mnist(config)


# In[12]:


def plotPCA(data,fileName,accuracy):
    pointColor = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
    dataArr = np.array(data)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(8):
        for j in range(20):
            index = 20*i + j
            s = "%.1f"%(accuracy[index]*100)
            plt.plot(dataArr[index,0], dataArr[index,1], c=pointColor[i])
            plt.text(dataArr[index,0], dataArr[index,1], s, color=pointColor[i])
    plt.savefig('report_data/'+fileName)
    plt.show()


# In[13]:


hidden_layer1 = np.asarray(hidden_layer1)
print(hidden_layer1.shape)
whole_model = np.asarray(whole_model)
print(whole_model.shape)
# do PCA
pca=PCA(n_components=2)
recon_hidden_layer1 = pca.fit_transform(hidden_layer1)
plotPCA(recon_hidden_layer1, "layer1.png",accuracy_record)
recon_whole_model = pca.fit_transform(whole_model)
plotPCA(recon_whole_model, "whole_model.png",accuracy_record)

