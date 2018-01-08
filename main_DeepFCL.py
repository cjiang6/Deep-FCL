# -*- coding: utf-8 -*-
"""
State representation learning from raw visual observation

Created on Mon Dec 11 10:10:18 2017

@author: jesse
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.slim as slim

class DeepFCL:
    
    def __init__(self, obs_dim, act_dim):
        
        # observation and state space dimension        
        self.obs_dim = obs_dim # height * width * channel
        self.act_dim = act_dim        
       
        # input variables
        self.obs_var = tf.placeholder(shape=[None,obs_dim*obs_dim*3], dtype=tf.float32, name="obs_var")
        self.goal_trj = tf.placeholder(shape=[None,2], dtype=tf.float32, name="goal_trj")
        self.is_training = tf.placeholder(shape=[],  dtype=tf.bool, name="train_cond")
        
        # ------- Define Observation-State Mapping Using Convolutional Network -----------------------
        
        # network parameters
        conv1_num = 16
        conv2_num = 32
        fc1_num = 256
        
        # resize the array of flattened input
        self.imageIn = tf.reshape(self.obs_var, shape=[-1,obs_dim,obs_dim,3])
        
        # convolutions acti: ReLU and spatial softmax
        self.conv1 = slim.conv2d(
            inputs=self.imageIn,num_outputs=conv1_num,kernel_size=[8,8],stride=[4,4],padding='VALID',biases_initializer=None)
        self.conv2 = slim.conv2d(
            inputs=self.conv1,num_outputs=conv2_num,kernel_size=[4,4],stride=[2,2],padding='VALID',biases_initializer=None)
#        self.conv3 = slim.conv2d(
#            inputs=self.conv2,num_outputs=conv3_num,kernel_size=[3,3],stride=[1,1],padding='VALID',biases_initializer=None)
#        self.conv4 = slim.conv2d(
#            inputs=self.conv3,num_outputs=conv4_num,kernel_size=[7,7],stride=[1,1],padding='VALID',biases_initializer=None)        
        
        # output layer
        self.convout = tf.concat([tf.contrib.layers.flatten(self.conv2), self.goal_trj], 1)                                                    
                                                           
        # Fully-connected
        self.W1 = tf.Variable(tf.random_normal([self.convout.get_shape().as_list()[1], fc1_num]))
        self.b1 = tf.Variable(tf.random_normal([fc1_num]))
        self.fc1 = tf.nn.relu(tf.matmul(self.convout, self.W1) + self.b1) 
        
        self.W2 = tf.Variable(tf.random_normal([fc1_num, act_dim]))
        self.b2 = tf.Variable(tf.random_normal([act_dim]))
        # output layer
        self.out = tf.nn.relu(tf.matmul(self.fc1, self.W2) + self.b2)
                
        # ------- Define Loss Function ---------------------------
        self.targetOut = tf.placeholder(shape=[None,2], dtype=tf.float32)
        self.out_error = tf.norm(self.targetOut - self.out, ord=2, axis=1) 
        
        # total loss
        self.loss = tf.reduce_mean(self.out_error)
        
        # Training Functions
        self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        self.train_op = self.optimizer.minimize(self.loss)
        
        
    def learn(self, observations, actions, goals, epi_start):
        
        # Prepre Training Data -------------------------------------------
        self.mean_obs = np.mean(observations, axis=0, keepdims=True)
        self.std_obs = np.std(observations, ddof=1)
        observations = (observations - self.mean_obs) / self.std.obs
        
        # number of samples in total
        num_samples = observations.shape[0] - 1
        
        # indices for all time steps where the episode continues
        indices = np.array([i for i in range(num_samples) if not epi_start[i + 1]], dtype='int32')
        np.random.shuffle(indices)

        # split indices into minibatches
        minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batchsize]))
                         for start_idx in range(0, num_samples - self.batchsize + 1, self.batchsize)]
        
        # Training -------------------------------------------------------
        init = tf.global_variables_initializer()
        num_epochs = 1000
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(init)
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                epoch_batches = 0
                enumerated_minibatches = list(enumerate(minibatchlist))
                np.random.shuffle(enumerated_minibatches)
                
                for i, batch in enumerated_minibatches:                    
                    _ , tmp_loss = sess.run([self.train_op,self.loss], feed_dict = {
                                                                        self.obs_var: observations[batch],
                                                                        self.goal_trj: goals[batch],
                                                                        self.targetOut: actions[batch],                                                                        
                                                                        self.is_training: True})
                    epoch_loss += tmp_loss
                    epoch_batches += 1
                    
                # print results for this epoch
                if (epoch+1) % 5 ==0:
                    print("Epoch {:3}/{}, loss:{:.4f}".format(epoch+1, num_epochs, epoch_loss / epoch_batches))                    
                
            # save the updated model
            saver.save(sess, "/tmp/model.ckpt")
            predicted_action = sess.run(self.out, feed_dict={self.obs_var: observations})
        plt.close("Learned Policy")
        
        return predicted_action
        
        
#    def test(self, observations):
#        observations = (observations - self.mean_obs) / self.std_obs
#        saver = tf.train.Saver()
#        with tf.Session() as sess:
#            # load the model and output action
#            saver.restore(sess, "tmp/model.ckpt")
#            act_output = sess.run(self.predict, feed_dict = {self.obs_var: observations})
#            
#        return act_output
        
        
if __name__ == '__main__':
    
    print('\nFormation Control Task\n')

    print('Loading and displaying training data ... ')
    training_data = np.load('training_data.npz')
    #plot_observations(training_data['observations'], name="Observation Samples (Subset of Training Data) -- Simple Navigation Task")

    print('Learning a policy ... ')
    fcl = DeepFCL(16 * 16 * 3, 2)
    training_states = fcl.learn(**training_data)
#    plot_representation(training_states, training_data['rewards'],
#                            name='Observation-State-Mapping Applied to Training Data -- Simple Navigation Task',
#                            add_colorbar=True)
    
        
        
        
        
        
