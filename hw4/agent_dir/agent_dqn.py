from agent_dir.agent import Agent
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np 
import matplotlib.pyplot as plt

class Agent_DQN(Agent):

    def __init__(self, env, args):
        self.args = args
        super(Agent_DQN,self).__init__(env)
        tf.reset_default_graph() 
        self.set_variable()
        self.build_model()
        
        
        if args.test_dqn or args.keep:
            print('loading trained model')
            self.saver.restore(self.sess, save_path = self.model_path)
        
    def train(self):
        #variables
        self.frame = 0
        self.episode = 0
        self.step = 0
        self.reward_record = [0]
        self.reward_recent = [0]
        self.record = deque()
        
        if not self.args.keep:
            self.sess.run(tf.global_variables_initializer())
            self.syncronize()
        
        while np.mean(self.reward_recent) < 500:
            if len(self.reward_recent) > 100:
                del self.reward_recent[0]
            
            #game start
            reward_total = self.play_game()
            
            #record a game result    
            self.reward_record.append(reward_total) 
            self.reward_recent.append(reward_total) 
            self.episode += 1
            
            
            if self.episode % 100 == 0:
                self.saver.save(self.sess, self.model_path)
            if self.episode % 10 == 0: 
                self.f.write(str(self.episode)) 
                self.f.write('  ')
                self.f.write(str(np.mean(self.reward_recent)))
                self.f.write('\n')                
                print('Game NO: ', self.episode,
                      'frames: ', self.frame,
                      'step: ',self.step,
                      'reward: ', reward_total,
                      'reward_arvg: ', np.mean(self.reward_recent))
            
    def play_game(self):
        state = self.env.reset()
        game_over = False
        self.step = 0
        reward_total = 0
   
        while not game_over:
            self.frame += 1
            self.step += 1      

            if self.esp > FINAL_EXPLORATION and self.frame > TRAIN_START:
                self.esp -=  (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION            
                     
            state_real = np.reshape(state, (1, 84, 84, 4))
            
            Q_output = self.sess.run(self.Q, feed_dict = {self.input : state_real})
            
            if self.esp > np.random.rand(1):
                action = np.random.randint(self.action_num)
            else:
                action = np.argmax(Q_output)
            
            state_next, reward, game_over, l = self.env.step(action+1) #state, reward, done,
            
            self.record.append((np.copy(state), np.copy(state_next), action ,reward, game_over))
            state = state_next
            reward_total += reward
            
            if len(self.record) > MEMORY_SIZE:
                self.record.popleft()
            
            #update model
            if self.frame > TRAIN_START and self.frame % ONLINE_UPDATE == 0:
                self.update_model_parameters()
                
        return reward_total                
    def update_model_parameters(self):
        state_stack = deque()
        action_stack = deque()
        reward_stack = deque()
        game_over_stack = deque()
        state_next_stack = deque()
                    
        sample = random.sample(self.record, self.sample_batch)
        
        for state, state_next, action, reward, game_over in sample:
            state_stack.append(state)
            state_next_stack.append(state_next)
            action_stack.append(action)
            reward_stack.append(reward)
            game_over_stack.append(game_over)
        
        
        Q_s_output = self.sess.run( self.Q_s, feed_dict={self.input: np.array(state_next_stack)})
            
        expected_values = reward_stack + (1 - (np.array(game_over_stack)+0)) * DISCOUNT * np.max(Q_s_output, axis=1)

        self.sess.run(self.train_op,feed_dict={self.input: np.array(state_stack),self.action: action_stack, 
            self.expected_value: expected_values})
         
        if self.frame % SYNCHRONIZE == 0:
            self.syncronize()#同步Q network
    
    def set_variable(self):
        self.action_num = 3
        self.sample_batch = 32
        self.model_path = "./model_file/dqn/dqn-0"
        self.f = open('reward_dqn_1.txt', 'w')
        if self.args.keep:
            self.esp = 0.01
        else:
            self.esp = 1
        
        global MEMORY_SIZE, DISCOUNT, SYNCHRONIZE, LEARNING_RATE, TRAIN_START, ONLINE_UPDATE
        global FINAL_EXPLORATION, START_EXPLORATION, EXPLORATION
        
        MEMORY_SIZE = 10000
        DISCOUNT = 0.99
        SYNCHRONIZE = 1000
        LEARNING_RATE = 0.0001
        TRAIN_START = 10000
        ONLINE_UPDATE = 4
        FINAL_EXPLORATION = 0.05
        START_EXPLORATION = 1.
        EXPLORATION = 1000000
        
    def build_model(self):
        self.input = tf.placeholder("float", [None, 84, 84, 4])
        self.action= tf.placeholder(tf.int64, [None])
        self.expected_value = tf.placeholder(tf.float32, [None], name='expected_value')

        self.f1 = tf.get_variable("f1", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f2 = tf.get_variable("f2", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f3 = tf.get_variable("f3", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        self.w1 = tf.get_variable("w1", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
        self.w2 = tf.get_variable("w2", shape=[512, self.action_num], initializer=tf.contrib.layers.xavier_initializer())
        
        self.f1_s = tf.get_variable("f1_s", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f2_s = tf.get_variable("f2_s", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f3_s = tf.get_variable("f3_s", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        self.w1_s = tf.get_variable("w1_s", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
        self.w2_s = tf.get_variable("w2_s", shape=[512, self.action_num], initializer=tf.contrib.layers.xavier_initializer())

        self.Q = self.chain_tensor(self.input, self.f1, self.f2, self.f3 , self.w1, self.w2)
        self.Q_s = self.chain_tensor(self.input, self.f1_s, self.f2_s, self.f3_s , self.w1_s, self.w2_s)
        
        a_one_hot = tf.one_hot(self.action, self.action_num, 1.0, 0.0)
        self.Q_value = tf.reduce_sum(tf.multiply(self.Q, a_one_hot), reduction_indices=1)
        
        #update parameters
        error = tf.abs(self.expected_value - self.Q_value)
        error = self.clipped_error(error)
        self.loss = tf.reduce_mean(tf.reduce_sum(error))

        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE,momentum=0,epsilon= 1e-8, decay=0.99)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.saver = tf.train.Saver(max_to_keep=None)
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
        config.gpu_options.allow_growth = True #allocate dynamically
        self.sess = tf.InteractiveSession(config=config)
    
    def chain_tensor(self, input, f1, f2, f3 , w1, w2):
        c1 = tf.nn.relu(tf.nn.conv2d(input, f1, strides=[1, 4, 4, 1],data_format="NHWC", padding = "VALID"))
        c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1],data_format="NHWC", padding="VALID"))
        c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1,1,1,1],data_format="NHWC", padding="VALID"))

        l1 = tf.reshape(c3, [-1, w1.get_shape().as_list()[0]])

        l2 = self.lrelu(tf.matmul(l1, w1))
        output = tf.matmul(l2, w2)

        return output       
    def make_action(self, observation, test=True):
        state = np.reshape(observation, (1, 84, 84, 4))
        Q_output = self.sess.run(self.Q, feed_dict = {self.input : state})

        self.esp = 0.01

        if self.esp > np.random.rand(1):
            action = np.random.randint(self.action_num)
        else:
            action = np.argmax(Q_output)

        return action+1
    def syncronize(self):
        self.sess.run(self.w1_s.assign(self.w1))
        self.sess.run(self.w2_s.assign(self.w2))
        self.sess.run(self.f1_s.assign(self.f1))
        self.sess.run(self.f2_s.assign(self.f2))
        self.sess.run(self.f3_s.assign(self.f3))
    def lrelu(self, x, alpha = 0.01):
        return tf.maximum(x, alpha * x)
    def clipped_error(self, x): 
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        pass