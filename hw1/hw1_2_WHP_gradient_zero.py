
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from functions import *


# In[67]:


def shallow_model(x_data, y_data, batch_size, epoch):
    units = 3
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    data_len = len(x_data)
    x_data = np.reshape(x_data, [data_len, 1])
    y_data = np.reshape(y_data, [data_len, 1])
    # variable initializer
    tf.Variable.initializer.setter(tf.truncated_normal_initializer())
    # build tf.dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)) # data to tf.data.Dataset
    dataset_batch = dataset.shuffle(data_len) # shuffle batch data (for training)
    dataset_batch = dataset.batch(batch_size) # batch data (for training)
    dataset = dataset.batch(data_len) # hole source data as a batch (for prediction)
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes) # dataset iterator
    next_example, next_label = iterator.get_next() # next batch as input, output
    
    # define initialize iterator operation
    training_init_op = iterator.make_initializer(dataset_batch)
    prediction_init_op = iterator.make_initializer(dataset)
    
    # build hidden layers
    hidden_layer0 = tf.layers.dense(inputs=next_example, units=units+1, activation=tf.nn.relu)
    hidden_layer1 = tf.layers.dense(inputs=hidden_layer0, units=units, activation=tf.nn.relu)
    # output layer y
    y = tf.layers.dense(inputs=hidden_layer1, units=1)
    # define mse loss
    loss = tf.losses.mean_squared_error(labels = next_label, predictions = y)
    # define var as colloction of all variable
    var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0:3:2]
    # define gradients norm in var[0:5:2] (only get weight not bias)
    gradients = tf.gradients(loss, var)
    grad_flat0 = tf.reshape(gradients[0], [units+1,])
    grad_flat1 = tf.reshape(gradients[1], [units*(units+1),])
    grad_flat = tf.concat([grad_flat0, grad_flat1], 0)
    grad_norm = tf.norm(grad_flat)
    # optimizer 
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    # initialize model
    sess.run(tf.global_variables_initializer())
    min_norm = 1000
    grad_norm_value = 10
    # start training
    ep = 0
    while ((ep < epoch) or (grad_norm_value > 0.03 and ep < 5 * epoch)):
        # batch training
        sess.run(training_init_op)
        while True:
            try:
                sess.run([optimizer])
            except tf.errors.OutOfRangeError:
                break
        
        # calculate gradient norm
        sess.run(prediction_init_op)
        grad_norm_value, loss_value = sess.run([grad_norm, loss])
        ep += 1
        if ep % 10 == 0:
            print('%5d epoch'% ep,"loss: %2.8f"% loss_value, 'norm: %f'% grad_norm_value)
    print('%5d epoch'% ep,"loss: %2.8f"% loss_value, 'norm: %f'% grad_norm_value)
    print('start sampling...')
    # sample 1000 point beside this weight
    radius = 0.0000002
    sample_num = 500
    # store var in var 0
    sample_op = []
    for i, v in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0:5]):
        var0 = sess.run(tf.identity(v))
        bias = (tf.random_uniform(v.shape, minval=-1, maxval=1, dtype=tf.float64))
        sample_op.append(tf.assign(v, var0 + bias * radius))
    
    # predict and plot
    minimal_ratio = 0
    for i in range(sample_num):
        sess.run(prediction_init_op) # switch dataset from batch dataset to source dataset
        sess.run(sample_op)
        sample_loss = sess.run(loss) # predict y
        if sample_loss > loss_value:
            minimal_ratio += 1
    minimal_ratio = minimal_ratio/sample_num
    print('minimal_ratio', minimal_ratio)
    history = {'minimal_ratio':minimal_ratio, 'loss':loss_value}
    return history


# In[3]:


x = np.linspace(0,2,50000)
y = sine_wave(4, 1, x)
plt.plot(x, y)


# In[69]:


train_num = 100
history = [None]*train_num
for i in range(train_num):
    print('turn %d'%i)
    tf.reset_default_graph()
    history[i] = shallow_model(x, y, 1000, 150)


# In[84]:


his_list = [[],[]]
for d in history:
    his_list[0].append(d['loss'])
    his_list[1].append(d['minimal_ratio'])

plt.plot(his_list[1], his_list[0], marker='o', ls='')
plt.title('minimal ratio')
plt.xlabel('minimal ratio')
plt.ylabel('loss')
plt.savefig('report_data/minimal_ratio.png', dpi = 256)
plt.show()

