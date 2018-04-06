
# coding: utf-8

# In[2]:


from utils.models import Sequential
from utils.layers import Dense
from utils.CIFAR_10 import unpickle_all_file
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.backend.tensorflow_backend import set_session


# In[3]:


# get CIFAR-10 data
# train_data: {'data': (50000, 32 * 32), 'labels': (50000, 1)}
# train_data: {'data': (10000, 32 * 32), 'labels': (10000, 1)}
train_data, test_data = unpickle_all_file()
train_data['data'] = train_data['data'] / 255
test_data['data'] = test_data['data'] / 255


# In[13]:


def build_model(param, x_train, y_train, x_test, y_test):
    model = Sequential()
    # CNN
    model.add(Conv2D(filters = param // 3, kernel_size = (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = param // 3, kernel_size = (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters = (param // 3) * 2, kernel_size = (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = (param // 3) * 2, kernel_size = (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    
    
    # Dense
    model.add(Dense(units = param * 3))
    model.add(Activation('relu'))
    model.add(Dense(units = param * 3))
    model.add(Activation('relu'))
    model.add(Dense(units = 10))
    model.add(Activation('softmax'))
    
    return model


# In[14]:


import matplotlib.pyplot as plt
param_num = range(1, 101, 3)
total_params = [0] * 100
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
sess = tf.Session(config = session_config)
set_session(sess)
plt.figure(1)
plt.title('Loss')
plt.figure(2)
plt.title('Accuracy')
for i, v in enumerate(param_num):
    model = build_model(100, train_data['data'], train_data['labels'], test_data['data'], test_data['labels'])
    total_params[i] = model.count_params()
    print(model.count_params())
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data['data'], train_data['labels'], epochs=100, batch_size=1000, validation_data=(test_data['data'], test_data['labels']))
    plt.figure(1) # loss figure
    plt.plot(total_params[i], history.history['loss'][-1], 'bo') # plot training loss
    plt.plot(total_params[i], history.history['val_loss'][-1], 'go') # plot val loss
    plt.figure(2) # acc figure
    plt.plot(total_params[i], history.history['acc'][-1], 'bo') # plot training acc
    plt.plot(total_params[i], history.history['val_acc'][-1], 'go') # plot val acc
plt.figure(1)
plt.legend(['training','testing'])
plt.figure(2)
plt.legend(['training','testing'])

