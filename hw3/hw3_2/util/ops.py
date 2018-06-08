
# coding: utf-8
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import os


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)

def pixelShuffler(inputs, scale=2):
    size = inputs.get_shape().as_list()
    #batch_size = size[0]
    h = size[1]
    w = size[2]
    c = size[3]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [-1, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [-1, h * scale, w * scale, 1]
    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output