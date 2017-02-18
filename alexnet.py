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

from visualization.feature_optimization import optimize_feature, save_optimized_image_to_disk
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
from dataset_manager_ijcai import Sketch
import config
import latex_sketch as latex

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
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
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
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
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
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)

#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(net_data["fc8"][0])
fc8b = tf.Variable(net_data["fc8"][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)

"""Loss and training"""
cross_entropy = -tf.reduce_sum(y*tf.log(prob + 1e-9))
train_step = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(cross_entropy)

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
    if config.restore_file == '':
      print 'Restoring from ', ckpt.model_checkpoint_path
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print 'Restoring from ', os.path.join(config.restore_path, config.restore_file)
      saver.restore(sess, os.path.join(config.restore_path, config.restore_file))
else:
  ckpt = 0

"""Training"""
# ims = []
# truths = []
print 'Training set size: ', dataset.training_size
print 'Test set size', dataset.test_size
print 'Learning rate: ', config.learning_rate
print 'Partial Amount: ', config.partial_amount
if config.training:
  for j in range(0, config.epochs):
    for i in range(1, dataset.training_size+1):
      if not i%30:
        print 'Train Step: ', i

      """Next training batch"""
      im, truth = dataset.next_batch_respecting_classes(1)
      # im, truth = dataset.next_batch(1)

      """Run training step"""
      sess.run(train_step, feed_dict={x: im, y: truth})

      # expected_number = truth[0].index(1.0)
      # key = class_names[expected_number]
      # print key
      # imshow(im[0])

      # ims.append(im)
      # truths.append(truth)

      # output = sess.run(prob, feed_dict={x: im, y: truth})
      # inds = argsort(output)[0,:]

      # for i in range(5):
      #   # print expected[0].index(1.0)
      #   print class_names[inds[-1-i]], output[0, inds[-1-i]]
      # print
      # if i%57 == 0:
      #   print i/57, ' - Images from each class.'


"""Save model trained"""
if config.save_training:
  saver = tf.train.Saver(tf.all_variables())
  saver.save(sess, os.path.join(config.restore_path, 'PA_' + str(config.partial_amount) +'--LR_' + str(config.learning_rate) + '_' + str(config.iteration) + '_model.ckpt'), global_step=i)


"""Testing"""
correct = 0
correct_1 = 0
correct_train = 0
correct_train_1 = 0
correct_IN_5 = 0
correct_IN_1 = 0

dict_expected = {}
dict_correct_5 = {}
dict_correct_1 = {}
dict_false_positives_5 = {}
dict_false_positives_1 = {}

"""Used only in half-trained"""
used_train = [class_names[dataset.dataset[folder]] for folder in dataset.folders[0:38]]
print 'Used in training: ', used_train, len(used_train)

# for i in xrange(96):
#   opt_output = optimize_feature((227, 227, 3), x, conv1[:,:,:,i], sess)
#   save_optimized_image_to_disk(opt_output, "conv1_fine_" + str(i) + ".png")
# for i in xrange(256):
#   opt_output = optimize_feature((227, 227, 3), x, conv2[:,:,:,i], sess)
#   save_optimized_image_to_disk(opt_output, "conv2_fine_" + str(i) + ".png")
# for i in xrange(384):
#   opt_output = optimize_feature((227, 227, 3), x, conv3[:,:,:,i], sess)
#   save_optimized_image_to_disk(opt_output, "conv3_fine_" + str(i) + ".png")
# for i in xrange(384):
#   opt_output = optimize_feature((227, 227, 3), x, conv4[:,:,:,i], sess)
#   save_optimized_image_to_disk(opt_output, "conv4_fine_" + str(i) + ".png")
# for i in xrange(256):
#   opt_output = optimize_feature((227, 227, 3), x, conv5[:,:,:,i], sess)
#   save_optimized_image_to_disk(opt_output, "conv5_fine_" + str(i) + ".png")

# raise

if config.test:
  for j in range(0, dataset.test_size):
    """Next test image"""
    im, truth = dataset.next_test()

    output = sess.run(prob, feed_dict={x: im, y: [[0.0] * 1000]})

    inds = argsort(output)[0,:]

    expected_number = truth[0].index(1.0)

    """Class names in the outputs"""
    outs = []
    for i in range(5):
      outs.append(class_names[inds[-1-i]])
    # for i in range(5):
    #   print class_names[inds[-1-i]], output[0, inds[-1-i]]
    key = class_names[expected_number]
    if key not in dict_expected:
      dict_expected[key] = 1
    else:
      dict_expected[key] += 1

    """Verify correct answers"""
    if key in outs:
      #print 'correct 5: ', key, outs
      dict_correct_5[key] = dict_correct_5.get(key, 0) + 1
      correct += 1
    else:
      for l in outs:
        dict_false_positives_5[l] = dict_false_positives_5.get(l, 0) + 1
    if key in outs[0]:
      dict_correct_1[key] = dict_correct_1.get(key, 0) + 1
      #print 'correct 1: ', key, outs[0]
      correct_1 += 1
    else:
      dict_false_positives_1[outs[0]] = dict_false_positives_1.get(outs[0], 0) + 1



    # print 'Expected: ', class_names[expected_number]
    # print 'Expected class is in training: ', class_names[expected_number] in used_train
    if class_names[expected_number] in used_train:
      if class_names[expected_number] in outs:
        correct_train += 1
      if class_names[expected_number] in outs[0]:
        correct_train_1 += 1
    # imshow(im[0])
    # print

    # """FOR IMAGENET"""
    # inds = argsort(output)[1,:]
    # expected_number = truth[1].index(1.0)

    # """Class names in the outputs"""
    # outs = []
    # for i in range(5):
    #   outs.append(class_names[inds[-1-i]])

    # if key in outs:
    #     print 'correct 5 IN: ', key, outs
    #     correct_IN_5 += 1
    # if key in outs[0]
    #     print 'correct 1 IN: ', key, outs[0]
    #     correct_IN_1 += 1




    # answers.append(correct)

    # if not (j+1)%30:
    #   print 'Tested: ', str(j+1) + '. Top-5: ', str(float(correct)/float(j+1)) + '. Top-1: ', str(float(correct_1)/float(j+1))


#print dict_expected
if not os.path.isdir('results_ijcai_14_02_17'):
  os.mkdir('results_ijcai_14_02_17')
with open('results_ijcai_14_02_17/PA_' + str(config.partial_amount) + '--LR_' + str(config.learning_rate) + '--E_' + str(config.epochs) + '--' + str(config.iteration)  + '.txt', "wr") as fid:
  print >>fid, 'Training Size: ' + str(dataset.training_size)
  print >>fid, 'Epochs: ' + str(config.epochs)
  print >>fid, 'Learning Rate: ' + str(config.learning_rate)
  print >>fid, 'Partial Amount: ' + str(config.partial_amount)
  print >>fid, 'Test Size: ' + str(dataset.test_size)
  print >>fid, 'Top-5: ' + str(correct)
  print >>fid, 'Top-1: ' + str(correct_1)
  print >>fid, 'Trained Top-5: ' + str(correct_train)
  print >>fid, 'Trained Top-1: ' + str(correct_train_1)
  print >>fid, 'Not-Trained Top-5: ' + str(correct-correct_train)
  print >>fid, 'Not-Trained Top-1: ' + str(correct_1-correct_train_1)
  print >>fid, 'False Positives 5: ' + str(dict_false_positives_5)
  print >>fid, 'False Positives 1: ' + str(dict_false_positives_1)
  print 'Correct in top-5: ', correct
  print 'Correct in top-5 \%: ', float(correct)/float(dataset.test_size)
  print 'Correct in top-1: ', correct_1
  print 'Correct in top-1 \%: ', float(correct_1)/float(dataset.test_size)
  print 'Correct trained in top-5: ', correct_train
  print 'Correct trained in top-1: ', correct_train_1
  # print 'Correct in top-5 IN: ', correct_IN_5
  # print 'Correct in top-1 IN: ', correct_IN_1


# print dict_expected
# print dict_correct_1
# print dict_correct_5
# print dict_false_positives_1
# print dict_false_positives_5

# print 'Latex table of each class:'

# print latexFullSketch(dict_expected, dict_correct_5, dict_correct_1, dict_false_positives_5, dict_false_positives_1)
# print latex.latexHalfSketch(dict_expected, dict_correct_5, dict_correct_1, dict_false_positives_5, dict_false_positives_1, used_train)
# print latex.latexNoSketch(dict_expected, dict_correct_5, dict_correct_1, dict_false_positives_5, dict_false_positives_1)
