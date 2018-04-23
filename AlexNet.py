################################################################################
# Taken from Michael Guerzhoy and Davi Frossard, 2016
# AlexNet implementation in TensorFlow, with weights
# For details see: 
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
# With code from https://github.com/ethereon/caffe-tensorflow
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

# from numpy import *
import os
import time
import numpy as np
from scipy.misc import imread
from caffe_classes import class_names
import tensorflow as tf

train_x = np.zeros((1, 227,227,3)).astype(np.float32)

num_classes = 2 # 1000
train_y = np.zeros((1, num_classes))

test_y = np.zeros((1, num_classes))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

#In Python 3.5, change this to:
#net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
# net_data = np.load("bvlc_alexnet.npy").item()

''' Tommy: reading in TF records files '''
# data_path = 'train.tfrecords'  # address to save the hdf5 file

# with tf.Session() as sess:
#     feature = {'train/image': tf.FixedLenFeature([], tf.string),
#                'train/label': tf.FixedLenFeature([], tf.int64)}
#     # Create a list of filenames and pass it to a queue
#     filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
#     # Define a reader and read the next record
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     # Decode the record read by the reader
#     features = tf.parse_single_example(serialized_example, features=feature)
#     # Convert the image data from string back to the numbers
#     image = tf.decode_raw(features['train/image'], tf.float32)
    
#     # Cast label data into int32
#     label = tf.cast(features['train/label'], tf.int32)
#     # Reshape image data into the original shape
#     image = tf.reshape(image, [224, 224, 3])
    
#     # Any preprocessing here ...
    
#     # Creates batches by randomly shuffling tensors
#     images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)


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
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


################################################################################
# Constructing AlexNet layer-by-layer
################################################################################

# The input image
x = tf.placeholder(tf.float32, (None,) + xdim)


#conv1: First convolutional layer with 96 kernels of size 11 x 11
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


#conv2: Second convolutional layer with 256 kernels of size 5 x 5
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

#conv3: Third convolutional layer
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4: Fourth convolutional layer
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5: Fifth convolutional layer
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
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

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

################################################################################
# Initialize the network (can take a while):

y = tf.placeholder(tf.float32, (None,) + ydim)

init = tf.global_variables_initializer()
with tf.name_scope("loss"):
	# test_y?
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=prob)
    loss = tf.reduce_mean(xentropy, name="loss")
	
with tf.name_scope("accuracy"):
	accuracy = tf.metrics.accuracy(labels=tf.argmax(labels,0), predictions=tf.argmax(prob, 0))

optimizer = tf.train.MomentumOptimizer(0.1, momentum=0.9)
training_op = optimizer.minimize(loss)

train_dataset = TRAIN_DATASET ## CONSTRUCT THESE
test_dataset = TEST_DATASET

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(train_dataset.num_examples // batch_size):
            X_batch, y_batch = train_dataset.next_batch(batch_size)
            sess.run(training_op, feed_dict={training: True, x: X_batch, y: y_batch})
		
        acc_test = accuracy.eval(feed_dict={x: test_dataset.images, y: test_dataset.labels})
        print(epoch, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./lm_2class.ckpt")