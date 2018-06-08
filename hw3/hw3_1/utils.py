import matplotlib
matplotlib.use('Agg')

import numpy as np
from PIL import Image
import numpy as np
from numpy.random import random_sample
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import csv


class dataset():
    def __init__(self):
        self.data_row = []
        self.data_reg = []
        self.data_len = 0
        self.batch_index = 0
    
    def next_batch(self, size):
        if size > self.data_len:
            print('batch size can not greater than data length')
            return
        
        batch = self.data_reg[self.batch_index: self.batch_index+size]
        self.batch_index += size
        if self.batch_index >= self.data_len:
            self.batch_index = self.batch_index % self.data_len
            batch = batch + self.data_reg[0: self.batch_index]
        return batch
    
    def load_data(self, data_dir, verbose = 2000):
        i = 0
        while(True):
            img_name = str(i) + '.jpg'
            img_dir = data_dir + img_name
            try:
                img = Image.open(img_dir)
                i = i + 1
            except:
                print('done! total loaded data: ' + str(i))
                break
            img_arr = np.asarray(img)
            self.data_row.append(img_arr)
            self.data_reg.append(np.float32(img_arr) / np.float32(255))
            
            if (verbose != 0) and (i % verbose == 0):
                print(str(i) + ' data have been loaded...')
        self.data_len = len(self.data_row)
        return

    def random_sample(self, size, dtype = 'reg'):
        data = None
        if dtype == 'reg':
            data = np.array(self.data_reg)
        elif dtype == 'row':
            data = np.array(self.data_row)
        else:
            print('wrong data type!')
            return
        index = (random_sample(size) * self.data_len).astype(np.int)
        return data[index]
    
    def shuffle(self):
        seed = random.random()
        random.seed(seed)
        random.shuffle(self.data_row)
        random.seed(seed)
        random.shuffle(self.data_reg)
        
    
def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot_samples(samples, h = 4, w = 4, color_dim = 3, save = False, filename = None, folder_path = 'out/'):
    fig = plt.figure(figsize=(h, w))
    gs = gridspec.GridSpec(h, w)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        sample = np.clip(sample, 0, 1)
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    if save and filename:
        plt.savefig(folder_path + filename, bbox_inches='tight')
        plt.close(fig)
    elif not save:
        plt.show()
    return