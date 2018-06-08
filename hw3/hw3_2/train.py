
# coding: utf-8

import matplotlib.pyplot as plt
from util.ACGAN_resnet import ACGAN
from util.DataManager import *
import numpy as np
import sys

#train_img = np.load('Dataset/train_imgs_new.npy')
#train_att = np.load('Dataset/train_attributes_new.npy')

train_att, train_img = preproess(sys.argv[1],sys.argv[2])
model = ACGAN()
model.build_model()
model.train(train_img,train_att)