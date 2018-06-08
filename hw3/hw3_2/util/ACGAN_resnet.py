
# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import os
from util.ops import *


def generator_block(inputs, output_channel, stride, scope, train = True):
    with tf.variable_scope(scope):
        net = tc.layers.convolution2d(
                inputs, output_channel, [3, 3], [stride, stride],
                padding='same',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None
                )
        net = tf.layers.batch_normalization(net, training=train)
        net = tf.nn.relu(net)
        
        net = tc.layers.convolution2d(
                inputs, output_channel, [3, 3], [stride, stride],
                padding='same',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None
                )
        net = tf.layers.batch_normalization(net, training=train)
        net = net + inputs
    return net

def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
    with tf.variable_scope(scope):
        net = tc.layers.convolution2d(
                inputs, output_channel, [kernel_size, kernel_size], [stride, stride],
                padding='same',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=leaky_relu
                )
        net = tc.layers.convolution2d(
                inputs, output_channel, [kernel_size, kernel_size], [stride, stride],
                padding='same',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None
                )
        net = net + inputs
        net = leaky_relu(net)
    return net

class ACGAN():
    
    def __init__(self):
        self.x_dim = 64
        self.y_dim = 23
        self.z_dim = 100
        
        self.var_init = False
        self.batch_size = 64
        self.eps = 1e-8
        self.lr = 2e-4
        self.iteration = 50000
        
        self.G_sample = None
        self.C_real_loss = None
        self.C_fake_loss = None
        self.DC_loss = None
        self.GC_loss = None
        self.theta_D = None
        self.theta_G = None
        self.D_solver = None
        self.G_solver = None
        
        self.xs = tf.placeholder(tf.float32,[None,64,64,3])
        self.ys = tf.placeholder(tf.float32,[None,self.y_dim])
        self.zs = tf.placeholder(tf.float32,[None,self.z_dim])
        
        self.hair_color = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
                          'green hair', 'red hair', 'purple hair', 'pink hair',
                          'blue hair', 'black hair', 'brown hair', 'blonde hair']
        self.eyes_color = ['gray eyes', 'black eyes', 'orange eyes',
                           'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
                           'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
        self.test_condition = [[8,22],[8,19],[8,21],[4,22],[4,21]]
        self.test_tags = None
        
        # define session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)  
        
    def build_model(self):
                
        self.G_sample = self.build_G(self.zs, self.ys)
        D_real, C_real = self.build_D(self.xs)
        D_fake, C_fake = self.build_D(self.G_sample, reuse=True)

        # Cross entropy aux loss
        self.C_real_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=C_real, labels=self.ys),axis=1))
        self.C_fake_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=C_fake, labels=self.ys),axis=1))
        
        D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
        
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
        
        # GAN D loss
        D_loss = D_real_loss+D_fake_loss
        self.DC_loss = (25*D_loss+self.C_real_loss)

        # GAN's G loss
        self.GC_loss = (25*G_loss+self.C_fake_loss)
        
        # variable list
        self.theta_D = [v for v in tf.global_variables() if 'discriminator' in v.name]
        self.theta_G = [v for v in tf.global_variables() if 'generator' in v.name]

        self.D_solver = (tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.DC_loss, var_list=self.theta_D))
        self.G_solver = (tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.GC_loss, var_list=self.theta_G))
        
    def cross_entropy(self, logit, y):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y))

    def generate_test_tags(self):
        self.test_tags = np.zeros([25, self.y_dim])
        for i in range(5):
            for j in range(5):
                self.test_tags[i*5+j][self.test_condition[i][0]] = 1
                self.test_tags[i*5+j][self.test_condition[i][1]] = 1

    def build_G(self, z, c, train=True):
        
        s = self.x_dim
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        g_dim = self.x_dim
        c_dim = 3
        block_num = 16
        
        with tf.variable_scope('generator'):
            noise_vector = tf.concat([z, c], axis=1)

            net = tc.layers.fully_connected(
                noise_vector, g_dim*s8*s8,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )

            net = tf.layers.batch_normalization(net, training=train)
            net = tf.reshape(net, [-1, s8, s8, g_dim])
            net = tf.nn.relu(net)

            net_shortcut = net

            #### generator block part ####
            for i in range(1, block_num+1, 1):
                name_scope = 'g_block_%d'%(i)
                net = generator_block(net, 64, 1, name_scope, train=train)

            net =  tf.layers.batch_normalization(net, training=train)
            net = tf.nn.relu(net)

            net = net_shortcut + net

            net = tc.layers.convolution2d(
                net, 256, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None
                )
            net = pixelShuffler(net)
            net =  tf.layers.batch_normalization(net, training=train)
            net = tf.nn.relu(net)

            net = tc.layers.convolution2d(
                net, 256, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None
                )            
            net = pixelShuffler(net)
            net =  tf.layers.batch_normalization(net, training=train)
            net = tf.nn.relu(net)

            net = tc.layers.convolution2d(
                net, 256, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None
                )
            net = pixelShuffler(net)
            net =  tf.layers.batch_normalization(net, training=train)
            net = tf.nn.relu(net)

            net = tc.layers.convolution2d(
                net, 3, [9, 9], [1, 1],
                padding='same',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None
                )
            net = tf.nn.tanh(net)

            return net
    
    def build_D(self, inputs, reuse=False, train=True):
        
        s = self.x_dim
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        d_dim = 64
            
        with tf.variable_scope('discriminator') as discriminator:
            if reuse:
                discriminator.reuse_variables()
    
            with tf.variable_scope('input_stage'):
                net = tc.layers.convolution2d(
                    inputs, 32, [4, 4], [2, 2],
                    padding='same',
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    activation_fn=leaky_relu
                    )
            
            #### discriminator block part ####
            
            # stage 1
            net = discriminator_block(net, 32, 3, 1, 'd_block_1_a')
            net = discriminator_block(net, 32, 3, 1, 'd_block_1_b')
            net = tc.layers.convolution2d(
                    net, 64, [4, 4], [2, 2],
                    padding='same',
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    activation_fn=None
                    )
            
            # stage 2
            net = discriminator_block(net, 64, 3, 1, 'd_block_2_a')
            net = discriminator_block(net, 64, 3, 1, 'd_block_2_b')
            net = tc.layers.convolution2d(
                    net, 128, [4, 4], [2, 2],
                    padding='same',
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    activation_fn=leaky_relu
                    )

            # stage 3
            net = discriminator_block(net, 128, 3, 1, 'd_block_3_a')
            net = discriminator_block(net, 128, 3, 1, 'd_block_3_b')
            net = tc.layers.convolution2d(
                    net, 256, [3, 3], [2, 2],
                    padding='same',
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    activation_fn=leaky_relu
                    )
            
            # stage 4
            net = discriminator_block(net, 256, 3, 1, 'd_block_4_a')
            net = discriminator_block(net, 256, 3, 1, 'd_block_4_b')
            net = tc.layers.convolution2d(
                    net, 512, [3, 3], [2, 2],
                    padding='same',
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    activation_fn=leaky_relu
                    )

            # stage 5
            net = discriminator_block(net, 512, 3, 1, 'd_block_5_a')
            net = discriminator_block(net, 512, 3, 1, 'd_block_5_b')
            net = tc.layers.convolution2d(
                    net, 1024, [3, 3], [2, 2],
                    padding='same',
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    activation_fn=leaky_relu
                    )
            
            # classify
            net = tc.layers.flatten(net)
            out_gan = tf.layers.dense(
                        net, 1,
                        activation=None,
                        kernel_initializer=tf.contrib.layers.xavier_initializer()
                        )

            out_aux = tf.layers.dense(
                        net, 23,
                        activation=None,
                        kernel_initializer=tf.contrib.layers.xavier_initializer()
                        )

        return out_gan, out_aux
    
    def sample_z(self,shape):
        return np.random.uniform(-1., 1., size=shape)
    
    def generate(self, z, y):
        self.restore()
        return self.sess.run(self.G_sample, feed_dict = {self.zs: z, self.ys: y})
    
    def plot(self,samples):
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_aspect('equal')
            sample = np.array((sample*127.5)+127.5, dtype=np.uint8)
            plt.imshow(sample.reshape(64, 64, 3))

        return fig

    def train(self, xs, ys):
        if not self.var_init:
            self.sess.run(tf.global_variables_initializer())
            self.var_init = True
        
        # reshape xs
        data_len = len(xs)
        batch_offset = 0
        xs = (xs-127.5)/127.5
        # shuffle dataset
        permutation = np.random.permutation(xs.shape[0])
        xs = xs[permutation,:]
        ys = ys[permutation,:]
        
        self.generate_test_tags()
        # split data in to batches
        for i in range(self.iteration):
            if batch_offset + self.batch_size > data_len:
                batch_offset = 0
                # shuffle dataset
                permutation = np.random.permutation(xs.shape[0])
                xs = xs[permutation,:]
                ys = ys[permutation,:]
            else:
                zs = self.sample_z((self.batch_size, self.z_dim))
                _, DC_loss_curr, C_real_loss_curr = self.sess.run([self.D_solver, self.DC_loss, self.C_real_loss],
                                                                   feed_dict={self.xs: xs[batch_offset:batch_offset + self.batch_size], 
                                                                              self.ys: ys[batch_offset:batch_offset + self.batch_size], 
                                                                              self.zs: zs})
                for j in range(3):
                    _, GC_loss_curr, C_fake_loss_curr = self.sess.run([self.G_solver, self.GC_loss, self.C_fake_loss],
                                                                        feed_dict={self.xs: xs[batch_offset:batch_offset + self.batch_size],
                                                                                    self.ys: ys[batch_offset:batch_offset + self.batch_size], 
                                                                                    self.zs: zs})


            batch_offset = batch_offset+self.batch_size

            if i % 50 == 0:
                print('Iter: {} D_loss: {:.4}; C_real: {:.4}; G_loss: {:.4}; C_fake: {:.4};'
                      .format(i, DC_loss_curr, C_real_loss_curr, GC_loss_curr, C_fake_loss_curr))
            if i % 300 == 0:
                samples = self.sess.run(self.G_sample, feed_dict={self.zs: self.sample_z((25, self.z_dim)), self.ys: self.test_tags})
                                
                fig = self.plot(samples)
                plt.savefig('samples/{}.png'.format(str(i).zfill(7)), bbox_inches='tight')
                #plt.show()
                plt.close(fig)
            if i%10000 == 0:
                self.save()

    def save(self, model_name = 'cgan'):
        model_file = os.getcwd() + '/model_file/' + model_name + '.ckpt'
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir('model_file')
        saver = tf.train.Saver()
        saver.save(self.sess, model_file)
        return

    def restore(self, model_name = 'cgan'):
        try:
            model_file = os.getcwd() + '/hw3_2/model_file/' + model_name + '.ckpt'
            print(model_file)
            if os.path.isdir(os.path.dirname(model_file)):
                saver = tf.train.Saver()
                saver.restore(self.sess, model_file)
                self.var_init = True
        except:
            pass
        return


