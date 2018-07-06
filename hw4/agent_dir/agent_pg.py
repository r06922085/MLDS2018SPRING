from agent_dir.agent import Agent
import scipy
import numpy as np
import gym
import tensorflow as tf
import os


# Action values to send to gym environment to move paddle up/down
UP_ACTION = 2
DOWN_ACTION = 3
# Mapping from action values to outputs from the policy network
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}

def prepro(I):
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1    # everything else (paddles, ball) just set to 1
    return np.expand_dims(I.astype(np.float32),axis=2)

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        # hyperparameters
        self.n_obs = [80, 80,1]           # dimensionality of observations
        self.learning_rate = 1e-3
        self.gamma = .99               # discount factor for reward
        self.decay = 0.99              # decay rate for RMSProp gradients
        self.hidden_size = 256
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        self.save_path='model_file/pg/pong.ckpt'
        self.obs = tf.placeholder(tf.float32, [None, *self.n_obs])
        self.actions = tf.placeholder(tf.float32, [None, 1])
        self.advantages = tf.placeholder(tf.float32, [None, 1])
        self.env = env

        self.build()

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.load_checkpoint()

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.first_move = True


    def build(self):
        net = tf.contrib.layers.flatten(self.obs)
        h = tf.layers.dense(
            net,
            units=self.hidden_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.action_prob = tf.layers.dense(
            h,
            units=1,
            activation=tf.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.loss = tf.losses.log_loss(
                    labels=self.actions,
                    predictions=self.action_prob,
                    weights=self.advantages)

        self.optimizer = tf.train.AdamOptimizer(0.001)
        gvs = self.optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -0.3, 0.3), var) for grad, var in gvs]
        self.train_op = self.optimizer.apply_gradients(capped_gvs)
        
        #self.train_op = self.optimizer.minimize(self.loss)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def discount_rewards(self, rewards, discount_factor):

        discounted_rewards = np.zeros_like(rewards)
        for t in range(len(rewards)):
            discounted_reward_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                discounted_reward_sum += rewards[k] * discount
                discount *= discount_factor
                if rewards[k] != 0:
                    
                    break
            discounted_rewards[t] = discounted_reward_sum

        return discounted_rewards


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################        
        n_episode = 0
        reward_record = []
        while True:
            self.pre_state = prepro(self.env.reset())
            a = self.env.action_space.sample()
            self.cur_state, _, _, _ = self.env.step(a)
            self.cur_state = prepro(self.cur_state)
            done = False
            total_reward = 0
            xs, ys, rs = [], [], []

            while not done:
                
                self.state_diff = self.cur_state - self.pre_state
                self.pre_state = self.cur_state

                a = self.choose_action(self.state_diff)
                self.cur_state, reward, done, _ = self.env.step(a)
                self.cur_state = prepro(self.cur_state)

                total_reward += reward

                xs.append(self.state_diff)
                ys.append(action_dict[a])
                rs.append(reward)

            n_episode += 1
            n_step = len(xs)

            # reward discount
            rs = np.array(rs)
            rs = self.discount_rewards(rs, self.gamma)
            rs -= np.mean(rs)
            rs /= np.std(rs)

            xs = np.array(xs)
            ys = np.expand_dims(np.array(ys),axis=1)
            rs = np.expand_dims(np.array(rs),axis=1)


            feed = {
                self.obs: xs,
                self.actions: ys,
                self.advantages: rs
            }

            _, loss_value = self.sess.run([self.train_op, self.loss],feed)
            reward_record.append(total_reward)
            if n_episode % 1 == 0:
                print('ep: %d reward: %.4f loss: %.8f time_step: %d'%(n_episode,total_reward,loss_value,n_step))
                
            if n_episode % 1000 == 0:    
                self.save(step=n_episode)
                print('SAVED MODEL: %d'%n_episode)
                #np.save('curves/pg_reward_record',reward_record)
                


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        if self.first_move:
            self.pre_state = prepro(observation)
            action = self.env.action_space.sample()
            self.first_move = False
        else:
            self.cur_state = prepro(observation)
            self.state_diff = self.cur_state-self.pre_state
            self.pre_state = self.cur_state

            feed = {
                self.obs: np.reshape(self.state_diff, [-1, *self.n_obs])
            }
            
            action_prob = self.sess.run(self.action_prob, feed)
            if 0.5 < action_prob:
                action = UP_ACTION
            else:
                action = DOWN_ACTION

        return action

    def choose_action(self, observations):
        observations = np.reshape(observations, [-1, *self.n_obs])
        feed = {
            self.obs: observations
        }
        
        action_prob = self.sess.run(self.action_prob, feed)
        if np.random.uniform() < action_prob:
            action = UP_ACTION
        else:
            action = DOWN_ACTION
        return action

    def save(self, step):
        print('Saving checkpoint...')
        self.saver.save(self.sess, self.save_path)

    def load_checkpoint(self):
        print('Loading checkpoint')
        self.saver.restore(self.sess, self.save_path)