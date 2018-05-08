import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
import os
from tensorflow.python.layers.core import Dense

class Seq2Seq():
    def __init__(self, voc_size, batch_size = 200, mode = 'train', beam = True, dtype = tf.float32):
        self.var_init = False
        
        # model parameter
        self.dtype = dtype
        self.input_timestep = 25
        self.dim_hidden = 512
        self.max_len = 25
        self.voc_size = voc_size
        self.lstm_num_layer = 3
        self.mode = mode
        self.beam = beam
        # model batch size (a int tensor)
        self.batch_size = batch_size
        self.beam_size = 5
        # feed tensor
        self.xs = tf.placeholder(dtype = tf.int32, shape = [self.batch_size, self.input_timestep])
        self.ys = tf.placeholder(dtype = tf.int32, shape = [self.batch_size, self.max_len+1])
        
        # model iporttant tensor (will be initialize by calling compile)
        self.decoder_loss = None
        self.train_op = None
        self.prediction = None
        
        # define session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)  
        
    def compile(self):
        
        encoder_output, encoder_final_state = self.Encoder()
        decoder_output = self.Decoder(encoder_output, encoder_final_state)
        
        if self.mode == 'train':
            # compute model loss
            decoder_logits = decoder_output.rnn_output
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_logits, labels=self.ys[:,1:])
            self.decoder_loss = tf.reduce_mean(cross_entropy)
            # define train_op and initialize variable
            self.train_op = tf.train.AdamOptimizer(epsilon=1e-3).minimize(self.decoder_loss)
        elif self.mode == 'test':
            # predict tensor
            if self.beam == True:
                self.prediction = tf.transpose(decoder_output.predicted_ids,perm=[0,2,1])
            else:
                self.prediction = decoder_output.sample_id
 
    def fit(self, xs, ys, epoch):
        if not self.var_init:
            self.sess.run(tf.global_variables_initializer())
            self.var_init = True
        data_len = len(xs)
        # split data in to batches
        for ep in range(epoch):
            # shuffle dataset
            permutation = np.random.permutation(xs.shape[0])
            xs = xs[permutation,:]
            ys = ys[permutation,:]
            
            batch_offset = 0
            ep_loss = 0
            batch_run = 0
            while batch_offset + self.batch_size < data_len:
                _, batch_loss = self.sess.run([self.train_op, self.decoder_loss], 
                                              feed_dict = {self.xs : xs[batch_offset:batch_offset + self.batch_size],
                                                           self.ys : ys[batch_offset:batch_offset + self.batch_size]})
                batch_offset += self.batch_size
                ep_loss += batch_loss
                batch_run += 1
                if batch_run % 100 == 0:
                    print('Epoch: {}, batch run: {}, loss: {}'.format(ep+1,batch_run, batch_loss))
                    self.save()
            ep_loss /= batch_run
            print('Epoch: {}, loss: {}'.format(ep + 1, ep_loss))
   
    def predict(self, x):
        if not self.var_init:
            self.restore()
        index_list = self.sess.run(self.prediction, feed_dict = {self.xs : x})
        return index_list
  
    
    def get_a_cell(self, num_units):
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units)
        return rnn_cell
    
    def Encoder(self):
        with tf.variable_scope('embedding'):
            emb_w = tf.Variable(tf.truncated_normal(shape=[self.voc_size, self.dim_hidden],stddev=0.1))
        encoder_input = tf.nn.embedding_lookup(emb_w, self.xs)
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.get_a_cell(self.dim_hidden) for _ in range(self.lstm_num_layer)])
        with tf.variable_scope('encoder'):
            output, final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_input, 
                                                    dtype=self.dtype, time_major=False, 
                                                    sequence_length=[self.input_timestep]*self.batch_size)
        return output, final_state
    
    def Decoder(self, encoder_output, encoder_final_state):
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            emb_w = tf.get_variable("embedding", shape = [self.voc_size, self.dim_hidden])
         
        if self.mode == 'test' and self.beam == True:
            print("use beamsearch decoding..")
            encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, multiplier=self.beam_size)
            encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=self.beam_size)
        
        attention_output = tf.contrib.seq2seq.LuongAttention(self.dim_hidden,encoder_output)
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.get_a_cell(self.dim_hidden) for _ in range(self.lstm_num_layer)])
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_output, 
                                                               attention_layer_size= self.dim_hidden)
        projection_layer = Dense(self.voc_size, use_bias=False)
        
        if self.mode == 'train':
            decoder_input = tf.nn.embedding_lookup(emb_w, self.ys[:,:-1])
            decoder_seq_length = [self.input_timestep] * self.batch_size
            decoder_init_state = decoder_cell.zero_state(self.batch_size,self.dtype).clone(cell_state=encoder_final_state)
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(decoder_input, decoder_seq_length, 
                                                                         emb_w, 0.2, time_major=False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_init_state, 
                                                               output_layer=projection_layer)

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder, 
                                                              maximum_iterations=self.max_len)
        elif self.mode == 'test':
            start_tokens = tf.ones([self.batch_size], tf.int32)
            end_token = 2
            if self.beam == True:
                decoder_init_state = decoder_cell.zero_state(self.batch_size*self.beam_size, 
                                                             self.dtype).clone(cell_state=encoder_final_state)
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=emb_w,
                                                                                 start_tokens=start_tokens, end_token=end_token,
                                                                                 initial_state=decoder_init_state,
                                                                                 beam_width=self.beam_size,
                                                                                 output_layer=projection_layer)
            else:
            
                decoder_init_state = decoder_cell.zero_state(self.batch_size,self.dtype).clone(cell_state=encoder_final_state)
                decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=emb_w,
                                                                           start_tokens=start_tokens, end_token=end_token)
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                                                                            initial_state=decoder_init_state,
                                                                            output_layer=projection_layer)
            
                                                                                 
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                    maximum_iterations=self.max_len)
           
        return outputs
    def save(self, model_name = 'chatbot'):
        model_file = os.getcwd() + '/train_model_file/' + model_name + '.ckpt'
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir('model')
        saver = tf.train.Saver()
        saver.save(self.sess, model_file)
        return

    def restore(self, model_name = 'chatbot'):
        try:
            model_file = os.getcwd() + '/test_model_file/' + model_name + '.ckpt'
            if os.path.isdir(os.path.dirname(model_file)):
                saver = tf.train.Saver()
                saver.restore(self.sess, model_file)
                self.var_init = True
        except:
            pass
        return

