from model import GAN
from utils import dataset, plot_samples, sample_z
import tensorflow as tf
import numpy as np
import os
import random

bs = 64

def main():
    # set GPU card
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # load anime face
    data_dir = '../anime_face/data_64/images/'
    data_extra_dir = '../anime_face/extra_data/images/'
    ds = dataset()
    ds.load_data(data_dir, verbose = 0)
    ds.load_data(data_extra_dir, verbose = 0)
    ds.shuffle()

    # reset graph
    tf.reset_default_graph()

    # set session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)

    # build model
    model = GAN(sess, gf_dim=128)
    
    # training
    z_plot = sample_z(36, 100)
    
    # initial fake image
    z = sample_z((bs), 100)
    i = 1
    while True:
        if (i == 1) or (i <= 100 and i % 20 == 0) or (i <= 200 and i % 50 == 0) or (i <= 1000 and i % 100 == 0) or (i % 200 == 0):
            g_samples = model.generate(z_plot)
            plot_samples(g_samples, save = True, filename = str(i), folder_path = 'out2/', h=6, w=6)

        # train discriminator more
        for _ in range(5):
            real_img = ds.next_batch(bs)
            z = sample_z(bs, 100)
            fake_img = model.generate(z)
            # train D
            D_loss = model.train_D(real_img, fake_img)
            
        G_loss = model.train_G(bs)
        
        if (i % 100) == 0:
            model.save(model_name = 'WGAN_v2')
            z_loss = sample_z(64, 100)
            g_loss = model.generate(sample_z(32, 100))
            g, d = model.sess.run([model.G_loss, model.D_loss], 
                                  feed_dict = {model.xs: ds.random_sample(32), model.gs: g_loss, model.zs: z_loss})
            print(str(i) + ' iteration:')
            print('D_loss:', d)
            print('G_loss:', g, '\n')
            
        i = i + 1

if __name__ == '__main__':
    main()