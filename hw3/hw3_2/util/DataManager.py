
# coding: utf-8

# In[1]:


import numpy as np
import csv
import os
from PIL import Image


# In[2]:


hair_color = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
              'green hair', 'red hair', 'purple hair', 'pink hair',
              'blue hair', 'black hair', 'brown hair', 'blonde hair']
eyes_color = ['gray eyes', 'black eyes', 'orange eyes',
              'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
              'green eyes', 'brown eyes', 'red eyes', 'blue eyes']


# In[3]:


def att2one_hot(hair_index,eyes_index,att_len=23):
    attribute = np.zeros(att_len)
    attribute[hair_index] = 1
    attribute[eyes_index+12] = 1
    return attribute

def crop_center(img,cropx,cropy):
    y,x,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]

def preproess(filename,imgdir):

    with open(filename, 'r') as f:
        data_attribute = []
        data_img = []
        for idx, row in enumerate(csv.reader(f)):
            tags = row[1].split(' ')
            hair_index = hair_color.index(tags[0]+' hair')
            eyes_index = eyes_color.index(tags[2]+' eyes')

            # load original images
            img_path = os.path.join(imgdir, '{}.jpg'.format(idx))
            img = Image.open(img_path)
            img = img.resize((64,64),Image.ANTIALIAS)
            tmp_img = img
            img = np.array(img)
            data_attribute.append(att2one_hot(hair_index,eyes_index))
            data_img.append(img)

            # data augmented by flip images
            img_flip = np.fliplr(img)
            data_attribute.append(att2one_hot(hair_index,eyes_index))
            data_img.append(img_flip)

            # data augmented by rotate degree 5
            img_p5 = tmp_img.rotate(5)
            img_p5 = img_p5.resize((int(64*1.15),int(64*1.15)),Image.ANTIALIAS)
            img_p5 = np.array(img_p5)
            img_p5 = crop_center(img_p5, 64,64)
            data_attribute.append(att2one_hot(hair_index,eyes_index))
            data_img.append(img_p5)

            # data augmented by rotate degree -5
            img_m5 = tmp_img.rotate(-5)
            img_m5 = img_m5.resize((int(64*1.15),int(64*1.15)),Image.ANTIALIAS)
            img_m5 = np.array(img_m5)
            img_m5 = crop_center(img_m5, 64,64)
            data_attribute.append(att2one_hot(hair_index,eyes_index))
            data_img.append(img_m5)


        train_attributes = np.array(data_attribute)
        train_imgs = np.array(data_img)
        
        return train_attributes, train_imgs