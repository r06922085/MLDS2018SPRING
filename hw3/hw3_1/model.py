import tensorflow as tf
import tensorflow.contrib.layers as tcl
from ops import *
from utils import sample_z
import os

# GAN
class GAN():
    def __init__(self, sess, img_height=64, img_width=64, z_dim=100, gf_dim=64, df_dim=64, color_dim=3, dtype=tf.float32, init=True):
        self.sess = sess
        self.dtype = dtype
        
        self.color_dim = color_dim
        self.img_height = img_height
        self.img_width = img_width
        self.img_size = img_height * img_width * color_dim
        
        self.z_dim = z_dim
        self.zs = tf.placeholder(dtype = dtype, shape = [None, z_dim])
        self.xs = tf.placeholder(dtype = dtype, shape = [None, img_height, img_width, color_dim])
        self.gs = tf.placeholder(dtype = dtype, shape = [None, img_height, img_width, color_dim])
        
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        
        self.G_out = None
        self.G_img = None
        self.theta_D = None
        self.theta_G = None
        self.D_solver = None
        self.G_solver = None
        self.D_clip = None
        
        self.build(init)
        
    def build(self, init = True):
        self.G_out = self.build_G(self.zs)
        
        G_fake = self.build_D(self.G_out, False)
        D_real = self.build_D(self.xs, True)
        D_fake = self.build_D(self.gs, True)
        
        self.G_loss = tf.reduce_mean(G_fake)
        self.D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
        
        # gradient penalty
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.xs + (1 - epsilon) * self.gs
        d_hat = self.build_D(x_hat, True)
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * 10.)
        
        self.D_loss = self.D_loss + ddx
        
        self.theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.D_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9)\
                                    .minimize(self.D_loss, var_list=self.theta_D)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9)\
                                    .minimize(self.G_loss, var_list=self.theta_G)
            
        self.G_img = tf.reshape(self.G_out, [-1, self.img_height, self.img_width, self.color_dim], name='G_img')
    
        if init:
            initializer = tf.global_variables_initializer()
            self.sess.run(initializer)
        return
    
    def train_G(self, batch_size, loop = 1):
        for i in range(loop):
            _, loss = self.sess.run([self.G_solver, self.G_loss],
                                    feed_dict = {self.zs: sample_z(batch_size, self.z_dim)})
        return loss
    
    def train_D(self, real_img, fake_img, loop = 1):
        for i in range(loop):
            _, loss = self.sess.run([self.D_solver, self.D_loss], 
                                    feed_dict = {self.xs: real_img,
                                                 self.gs: fake_img})
        return loss
    
    def generate(self, z):
        return self.sess.run(self.G_out, feed_dict = {self.zs: z})
    
    def build_G(self, z):
        with tf.variable_scope("generator") as scope:
            h0_h = self.img_height // 16
            h0_w = self.img_width // 16
            
            bs = tf.shape(z)[0]
            fc = tcl.fully_connected(z, h0_h * h0_w * (self.gf_dim*8), activation_fn=tf.identity)
            conv1 = tf.reshape(fc, [-1, h0_h, h0_w, self.gf_dim*8])
            conv2 = tcl.conv2d_transpose(
                conv1, (self.gf_dim*4), [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv3 = tcl.conv2d_transpose(
                conv2, (self.gf_dim*2), [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv4 = tcl.conv2d_transpose(
                conv3, (self.gf_dim*1), [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv5 = tcl.conv2d_transpose(
                conv4, (self.color_dim), [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.tanh
            )
            return conv5
    
    
    def build_D(self, x, reuse):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
                
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, self.img_height, self.img_width, self.color_dim])
            conv1 = tcl.conv2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu
            )
            conv2 = tcl.conv2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv3 = tcl.conv2d(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv4 = tcl.conv2d(
                conv3, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv4 = tcl.flatten(conv4)
            fc = tcl.fully_connected(conv4, 1, activation_fn=tf.identity)
            return fc
        
    def save(self, model_name = 'WGAN'):
        model_file = os.getcwd() + '/model_file/' + model_name + '.ckpt'
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir('model')
        saver = tf.train.Saver()
        saver.save(self.sess, model_file)
        return

    def restore(self, model_path):
        try:
            model_file = model_path + '.ckpt'
            if os.path.isdir(os.path.dirname(model_file)):
                saver = tf.train.Saver()
                saver.restore(self.sess, model_file)
                self.var_init = True
        except:
            pass
        return