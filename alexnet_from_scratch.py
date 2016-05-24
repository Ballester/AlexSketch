################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imshow
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import tensorflow as tf

from caffe_classes import class_names
from alexnet_sketch_aux import Sketch
import config

# train_x = zeros((1, 227,227,3)).astype(float32)
# train_y = zeros((1, 1000))
# xdim = train_x.shape[1:]
# ydim = train_y.shape[1]



################################################################################
#Read Image

# x_dummy = (random.random((1,)+ xdim)/255.).astype(float32)
# i = x_dummy.copy()
# i[0,:,:,:] = imresize((imread("poodle.png")[:,:,:3]).astype(float32), (227, 227, 3))
# i = i-mean(i)

# i = tf.Variable("float")
# tf.reshape(i, [-1, 227, 227, 3])

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))


net_data = load("bvlc_alexnet.npy").item()

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape):  
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())


dataset = Sketch()
# im, truth = dataset.next_batch(1)
x = tf.placeholder("float", shape=[1, 227, 227, 3])
y = tf.placeholder("float", shape=[1, 1000])


#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4

conv1W = weight_variable(shape=net_data["conv1"][0].shape, name="conv1w")
conv1b = bias_variable(shape=net_data["conv1"][1].shape)

conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2


conv2W = weight_variable(shape=net_data["conv2"][0].shape, name="conv2w")
conv2b = bias_variable(shape=net_data["conv2"][1].shape)

conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1

conv3W = weight_variable(shape=net_data["conv3"][0].shape, name="conv3w")
conv3b = bias_variable(shape=net_data["conv3"][1].shape)

conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2

conv4W = weight_variable(shape=net_data["conv4"][0].shape, name="conv4w")
conv4b = bias_variable(shape=net_data["conv4"][1].shape)

conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)

#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2

conv5W = weight_variable(shape=net_data["conv5"][0].shape, name="conv5w")
conv5b = bias_variable(shape=net_data["conv5"][1].shape)

conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = weight_variable(shape=net_data["fc6"][0].shape, name="fc6w")
fc6b = bias_variable(shape=net_data["fc6"][1].shape)

fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = weight_variable(shape=net_data["fc7"][0].shape, name="fc7w")
fc7b = bias_variable(shape=net_data["fc7"][1].shape)

fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = weight_variable(shape=net_data["fc8"][0].shape, name="fc8w")
fc8b = bias_variable(shape=net_data["fc8"][1].shape)

fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)

"""Loss and training"""
cross_entropy = -tf.reduce_sum(y*tf.log(prob + 1e-9))
train_step = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

"""Initializing tensorflow variables and saver"""
saver = tf.train.Saver(tf.all_variables())
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

"""Create Model folder"""
if not os.path.isdir(config.restore_path):
  os.mkdir(config.restore_path)

"""Restore last model trained"""
if config.restore_last:
  ckpt = tf.train.get_checkpoint_state(config.restore_path)
  if ckpt.model_checkpoint_path:
    print 'Restoring from ', ckpt.model_checkpoint_path  
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
  ckpt = 0

"""Training"""
# ims = []
# truths = []
for k in range(0, 20):
  if config.training:
    for i in range(0, dataset.training_size):
      if not i%30:
        print 'Train Step: ', i

      """Next training batch"""
      im, truth = dataset.next_batch(1)

      """Run training step"""
      sess.run(train_step, feed_dict={x: im, y: truth})

      # ims.append(im)
      # truths.append(truth)

      # output = sess.run(prob, feed_dict={x: im, y: truth})
      # inds = argsort(output)[0,:]

      # for i in range(5):
      #   # print expected[0].index(1.0)
      #   print class_names[inds[-1-i]], output[0, inds[-1-i]]
      # print

  # plt.ion()
  # ax = plt.gca()
  # ax.set_ylabel('Correct in top-5')
  # ax.set_xlabel('Images tested')

  """Testing"""
  correct = 0
  correct_1 = 0
  # answers = []
  # js = []
  # plt.grid()

  if config.test:
    for j in range(0, dataset.test_size):
      if j == 0:
        pass
        # ax.set_autoscale_on(True)
        # line, = ax.plot(js, answers)
      
      """Next test image"""
      im, truth = dataset.next_test()
      
      output = sess.run(prob, feed_dict={x: im, y:[[0.0] * 1000]})
      inds = argsort(output)[0,:]

      expected_number = truth[0].index(1.0)

      """Class names in the outputs"""
      outs = []
      for i in range(5):
        outs.append(class_names[inds[-1-i]])
      # for i in range(5):
      #   print class_names[inds[-1-i]], output[0, inds[-1-i]] 

      """Verify correct answers"""
      if class_names[expected_number] in outs:
        # print 'Correct 5'
        correct += 1
      if class_names[expected_number] == outs[0]:
        # print 'Correct 1'
        correct_1 += 1
      
      # print
      # imshow(im[0])

      # answers.append(correct)
      # js.append(j)
      
      if not (j+1)%30:
        print 'Tested: ', str(j+1) + '. Top-5: ', str(float(correct)/float(j+1)) + '. Top-1: ', str(float(correct_1)/float(j+1))

# line.set_xdata(js)
# line.set_ydata(answers)
# ax.relim()
# ax.autoscale_view(True,True,True)
# plt.draw()

# images = []
# gray = imresize((imread('aux/ant.jpg')[:,:,:]).astype(float32), (227, 227, 3))
# # image = zeros((227, 227, 3))
# # image[:,:,0] = image[:,:,1] = image[:,:,2] = gray

# image = gray
# # print image.shape
# # images[0,:,:,:] = image
# output = sess.run(prob, feed_dict={x: [image], y: [[0.0] * 1000]})
# inds = argsort(output)[0,:]

# for i in range(5):
#   print class_names[inds[-1-i]], output[0, inds[-1-i]]  

print 'Correct in top-5: ', correct

"""Save model trained"""
if config.save_training:
  saver = tf.train.Saver(tf.all_variables())
  saver.save(sess, config.restore_path + 'trained_' + str(dataset.training_size) + '_model.ckpt', global_step=i)
  # plt.savefig('result_trained')