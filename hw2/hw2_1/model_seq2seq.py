from utils.dataManager import DataManager
from models.hw2_1_S2VT_attention import Seq2Seq
import tensorflow as tf
import numpy as np

max_len = 40
dataset = DataManager()
dictionary = dataset.clean_train_dict
voc_size = dictionary.voc_size
dataset.BuildTrainableData(max_len = max_len)

tf.reset_default_graph()
model = Seq2Seq(voc_size = voc_size, max_len = max_len, dtype = tf.float32)
model.compile()

try:
    model.restore()
except:
    pass

train_x = []
train_y = []
for i, labels in enumerate(dataset.train_y):
    for j, label in enumerate(labels[:5]):
        train_x.append(dataset.train_x[i])
        train_y.append(label)

epoch = 0
while True:
    epoch += 1
    print('epoch', epoch)
    model.fit(train_x, train_y, 150, 1)
    model.save()