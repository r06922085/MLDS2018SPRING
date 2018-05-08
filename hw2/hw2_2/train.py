# coding: utf-8

from utils.dataManager import DataManager
from utils.Seq2Seq import Seq2Seq
import tensorflow as tf
import numpy as np
import os
import pickle
import sys

dataset = DataManager()

# load training data and label
dataset.LoadData(file_name=sys.argv[1])
train_data=np.asarray(dataset.data)
train_label=np.asarray(dataset.label)

# load dictionary
dict_file = open('data/dictionary.txt','rb')
word_list = pickle.load(dict_file)
voc_size = len(word_list)
dataset.train_dict.train_dict = word_list
dataset.train_dict.voc_size = voc_size
dictionary = dataset.train_dict

# define epoch num
epoch_num = 30
tf.reset_default_graph()
model = Seq2Seq(voc_size)
model.compile()
model.fit(train_data, train_label, epoch_num)
