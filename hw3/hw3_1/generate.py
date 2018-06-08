from model import GAN
from utils import dataset, plot_samples, sample_z
import tensorflow as tf
import numpy as np
import sys



def main():
    # set session
    sess = tf.Session()
    model = GAN(sess=sess, init=False, gf_dim=128)
    model.restore(model_path = 'hw3_1/model_file/WGAN_v2')

    z_plot = np.random.uniform(-1., 1., size=[25, 100])
    img = model.generate(z_plot)
    plot_samples(img, save = True, h=5, w=5, filename='gan', folder_path='samples/')

if __name__=='__main__':
    main()