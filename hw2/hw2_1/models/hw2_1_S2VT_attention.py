import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
import os

"""Usage
Declare a Seq2Seq instance : model = Seq2Seq(vac_size)
Compile model : model.compile()
Train model : model.fit(xs, ys, batch_size, epoch)
"""
class Seq2Seq():
    def __init__(self, voc_size, max_len, dtype = tf.float64):
        self.var_init = False
        # model parameter
        self.dtype = dtype
        self.input_timestep = 80
        self.input_shape = [4096]
        self.encoder_units = 512
        self.decoder_units = 1024
        self.max_len = max_len
        self.voc_size = voc_size
        
        # model batch size (a int tensor)
        self.batch_size = None
        
        # feed tensor
        self.xs = tf.placeholder(dtype = dtype, shape = [None, self.input_timestep] + self.input_shape)
        self.ys = tf.placeholder(dtype = tf.int32, shape = [None, self.max_len]) # label : ex.[1, 4, 5, 6, 200]
        
        # model important tensor (will be initialize by calling compile)
        self.encoder_outputs = None
        self.decoder_output = None
        self.decoder_loss = None
        self.attention_reg = None
        self.train_op = None
        self.prediction = None
        
        # for attention
        self.attention_W = None
        self.attention_b = None
        
        # for transform decoder output from dimension 1024 >> 512
        self.decoder_W = None
        self.decoder_b = None
        
        # embeddings
        self.emb_W = None
        self.emb_b = None
        
        # training data
        self.train_x = None
        self.train_y = None
        
        # define session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)  
        
    def compile(self):
        # initialize attention transform
        self.attention_W = tf.Variable(tf.truncated_normal([(self.decoder_units // 2) * (self.input_timestep + 1), 
                                                           self.input_timestep], stddev=1.0 / np.sqrt(self.encoder_units), 
                                                           dtype = self.dtype))
        self.attention_b = tf.Variable(tf.zeros([self.input_timestep], dtype = self.dtype))
        # initialize emb transform
        self.emb_W = tf.Variable(tf.truncated_normal([(self.decoder_units // 2), self.voc_size], 
                                                     stddev=1.0 / np.sqrt(self.decoder_units // 2), 
                                                     dtype = self.dtype))
        self.emb_b = tf.Variable(tf.zeros([self.voc_size], dtype = self.dtype))
        # decoder output transform
        self.decoder_W = tf.Variable(tf.truncated_normal([self.decoder_units, self.decoder_units // 2 ], 
                        stddev=1.0 / np.sqrt(self.encoder_units), dtype = self.dtype))
        self.decoder_b = tf.Variable(tf.zeros([self.decoder_units // 2 ], dtype = self.dtype))
        
        # connect all conponents (decoder, attention, decoder)
        self.batch_size = tf.shape(self.xs)[0]
        self.encoder_outputs, encoder_final_state = self.Encoder()
        self.decoder_output = self.Decoder(encoder_final_state, self.encoder_outputs[:,-1,:])
        
        # compute model loss
        decoder_output_flat = tf.reshape(self.decoder_output, [-1, self.decoder_units])
        decoder_output_transform_flat = tf.nn.xw_plus_b(decoder_output_flat, self.decoder_W, self.decoder_b)
        decoder_logits_flat = tf.add(tf.matmul(decoder_output_transform_flat, self.emb_W), self.emb_b)
        decoder_logits = tf.reshape(decoder_logits_flat, (self.batch_size, self.max_len, self.voc_size))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_logits, labels=self.ys)
        self.decoder_loss = tf.reduce_mean(cross_entropy)
        
        # predict tensor
        self.prediction = tf.argmax(decoder_logits_flat, 1)
        
        # define train_op and initialize variable
        self.train_op = tf.train.AdamOptimizer(epsilon=1e-4).minimize(self.decoder_loss)
        
    def fit(self, xs, ys, batch_size, epoch):
        if not self.var_init:
            self.sess.run(tf.global_variables_initializer())
            self.var_init = True
        data_len = len(xs)
        # split data in to batches
        for ep in range(epoch):
            batch_offset = 0
            ep_loss = 0
            batch_run = 0
            while batch_offset < data_len:
                _, batch_loss = self.sess.run([self.train_op, self.decoder_loss], 
                                              feed_dict = {self.xs : xs[batch_offset:batch_offset + batch_size],
                                                           self.ys : ys[batch_offset:batch_offset + batch_size]})
                
                batch_offset += batch_size
                ep_loss += batch_loss
                batch_run += 1
                print(batch_loss)
            ep_loss /= batch_run
            print('epoch {}, loss: {}'.format(ep + 1, ep_loss))
            
    def predict(self, x):
        if not self.var_init:
            self.sess.run(tf.global_variables_initializer())
            self.var_init = True
        index_list = self.sess.run(self.prediction, feed_dict = {self.xs : [x]})
        return index_list
        
    def Encoder(self):
        # a list that length is batch_size, every element refers to the time_steps of corresponding input
        inputs_length = tf.fill([tf.shape(self.xs)[0]], self.input_timestep)
        rnn_cell = LSTMCell(self.encoder_units)
        # use bidirectional rnn as encoder architecture
        (fw_outputs, bw_outputs), (fw_final_state, bw_final_state) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cell, cell_bw=rnn_cell, inputs=self.xs,
                                            sequence_length=inputs_length, dtype=self.dtype))
        # merge every forward and backward output as total output
        output = tf.add(fw_outputs, bw_outputs) / 2
        # merge every forward and backward final state as final state
        state_c = tf.concat([fw_final_state.c, bw_final_state.c], axis = 1)
        state_h = tf.concat([fw_final_state.h, bw_final_state.h], axis = 1)
        final_state = LSTMStateTuple(c = state_c, h = state_h)
        return output, final_state

    def Attention(self, decoder_output): 
        # select what encoder output to feed the decoder
        # combine encoder_outputs and decoder_output as x
        encoder_outputs_batch_flat = tf.reshape(self.encoder_outputs, 
                                                shape = [-1, self.input_timestep * self.encoder_units])
        x = tf.concat([encoder_outputs_batch_flat, decoder_output], axis = 1) # shape = [B, units * (timestep + 1)]
        
        weight_logits = tf.nn.xw_plus_b(x, self.attention_W, self.attention_b) # shape = [B + T, T]
        weight_logits = tf.nn.tanh(weight_logits)
        
        weight = tf.nn.softmax(weight_logits)
        weight = tf.expand_dims(weight, axis = 2)
        
        attention_output = tf.multiply(weight, self.encoder_outputs)
        attention_output = tf.reduce_sum(attention_output, axis = 1)
        return attention_output
    
    def Decoder(self, encoder_final_state, encoder_final_output):
        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:
                # initialization
                transformed_output = tf.random_uniform(shape = tf.shape(encoder_final_output), dtype = self.dtype)
                attention_output = self.Attention(transformed_output)
                next_input = tf.concat([transformed_output, attention_output], axis = 1)
                next_state = encoder_final_state
                emit_output = None
                next_loop_state = None
                elements_finished = False
            else:
                emit_output = cell_output
                transformed_output = tf.nn.xw_plus_b(cell_output, self.decoder_W, self.decoder_b)
                attention_output = self.Attention(transformed_output)
                next_input = tf.concat([transformed_output, attention_output], axis = 1)
                next_state = cell_state
                next_loop_state = None
                
            elements_finished = (time >= self.max_len)
            return (elements_finished, next_input, next_state, emit_output, next_loop_state)

        rnn_cell = LSTMCell(self.decoder_units)
        emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(rnn_cell, loop_fn)
        outputs = tf.transpose(emit_ta.stack(), [1, 0, 2]) # transpose for putting batch dimension to first dimension
        return outputs
    
    def save(self, model_name = 'attention'):
        model_file = os.getcwd() + '/model_file/' + model_name + '.ckpt'
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir('model')
        saver = tf.train.Saver()
        saver.save(self.sess, model_file)
        return

    def restore(self, model_name = 'attention'):
        try:
            model_file = os.getcwd() + '/model_file/' + model_name + '.ckpt'
            if os.path.isdir(os.path.dirname(model_file)):
                saver = tf.train.Saver()
                saver.restore(self.sess, model_file)
                self.var_init = True
        except:
            pass
        return