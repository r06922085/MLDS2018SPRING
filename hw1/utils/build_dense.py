import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def build_dense(depth = 3, units = [16, 16, 16]):
    model = Sequential()
    model.add(Dense(units[0], input_shape = (1,), activation = 'relu')) # input layer and 1 hidden layer
    # add (depth-1) hidden layers
    for i in range(1, depth):
        model.add(Dense(units[i], activation = 'relu'))
    model.add(Dense(1)) # output layer
    return model

def train(model, x, y, batch_size, epochs, verbose = 1, optimizer = 'adam'):
    # compule
    model.compile(loss='mse', optimizer=optimizer)  
    # start training
    train_history = model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return train_history