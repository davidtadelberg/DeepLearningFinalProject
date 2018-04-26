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
import tensorflow as tf

print("Starting job...")

num_classes = 50 # 1000
batch_size = 8

image_width = 512 # Images were resized to fit this width earlier during preprocessing

lr_newvars = 1e-3
lr_pretrained = 2e-4

num_epochs = 2
print_every = 10
save_every = 1000

#In Python 3.5, change this to:
net_data = np.load(open("/weights/bvlc_alexnet.npy", "rb"), encoding="latin1").item()
# net_data = np.load("bvlc_alexnet.npy").item()

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
x = tf.placeholder(tf.float32, shape=(None, image_width, image_width, 3))

#conv1: First convolutional layer with 96 kernels of size 11 x 11
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0], trainable=False)
conv1b = tf.Variable(net_data["conv1"][1], trainable=False)
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
conv2W = tf.Variable(net_data["conv2"][0], trainable=False)
conv2b = tf.Variable(net_data["conv2"][1], trainable=False)
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
conv3W = tf.Variable(net_data["conv3"][0], trainable=False)
conv3b = tf.Variable(net_data["conv3"][1], trainable=False)
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

with tf.variable_scope("pretrained"):
    #conv4: Fourth convolutional layer
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0], trainable=True) # Set to trainable
    conv4b = tf.Variable(net_data["conv4"][1], trainable=True) # Set to trainable
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

# End of original AlexNet

with tf.variable_scope("newvars"):
    #conv5: Fifth convolutional layer
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.get_variable("conv5W", shape=net_data["conv5"][0].shape, initializer=tf.contrib.layers.xavier_initializer())
    conv5b = tf.get_variable("conv5b", shape=net_data["conv5"][1].shape, initializer=tf.contrib.layers.xavier_initializer())
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #fc6
    #fc(4096, name='fc6')
    # fc6W = tf.Variable(net_data["fc6"][0], name="fc6W")
    # fc6b = tf.Variable(net_data["fc6"][1], name="fc6b")

    # Because original shapes were different, when creating FC layers we need to come up with the right shapes.
    flattened_units_maxpool5 = int(np.prod(maxpool5.get_shape()[1:]))
    fc6W = tf.get_variable("fc6W", shape=(flattened_units_maxpool5, net_data["fc7"][0].shape[0]), initializer=tf.contrib.layers.xavier_initializer())
    fc6b = tf.get_variable("fc6b", shape=(net_data["fc7"][0].shape[0],), initializer=tf.contrib.layers.xavier_initializer())
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, flattened_units_maxpool5]), fc6W, fc6b)

    #fc7
    #fc(4096, name='fc7')
    fc7W = tf.get_variable("fc7W", shape=(net_data["fc7"][0].shape[0], num_classes), initializer=tf.contrib.layers.xavier_initializer())
    fc7b = tf.get_variable("fc7b", shape=(num_classes,), initializer=tf.contrib.layers.xavier_initializer())
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    # #fc8
    # #fc(1000, relu=False, name='fc8')
    # fc8W = tf.Variable(net_data["fc8"][0], name="fc8W")
    # fc8b = tf.Variable(net_data["fc8"][1], name="fc8")
    # fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

    #prob
    #softmax(name='prob'))
    prob = tf.nn.softmax(fc7)

################################################################################
# Initialize the network (can take a while):

def read_preprocess(num_epochs):
    # Read in data from tfrecord files
    filenames = tf.train.match_filenames_once(os.path.join('/mock_data/', '*'))
    filename_queue = tf.train.string_input_producer(filenames,
        num_epochs=num_epochs, shuffle=True)

    reader = tf.TFRecordReader()

    _, serialized = reader.read(filename_queue)

    feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
               'image/class/label': tf.FixedLenFeature([], tf.int64)}

    features = tf.parse_single_example(serialized, features=feature)

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    label = tf.cast(features['image/class/label'], tf.int32)

    image = tf.reshape(image, [image_width, image_width, 3])

    return image, label

def setup_input_pipeline():
    single_image, single_label = read_preprocess(num_epochs)
    return tf.train.shuffle_batch([single_image, single_label],
        batch_size=batch_size,
        capacity=1000,
        min_after_dequeue=batch_size*2)

y = tf.placeholder(tf.int32, shape=(None,))

# init = tf.global_variables_initializer()

# create a saver
saver = tf.train.Saver()


with tf.name_scope("loss"):
    # test_y?
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=fc7)
    loss = tf.reduce_mean(xentropy, name="loss")
    
with tf.name_scope("accuracy"):
    accuracy, update_op = tf.metrics.accuracy(labels=y, predictions=tf.argmax(prob, 0))

batch = setup_input_pipeline()
global_step = tf.Variable(0, trainable=False, name='global_step')

# Define 2 different optimizers, to train the pretrained and the randomly initialized weights, respectively.
pretrained_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pretrained")
new_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "newvars")

pretrained_optimizer = tf.train.AdamOptimizer(learning_rate=lr_pretrained)
newvars_optimizer = tf.train.AdamOptimizer(learning_rate=lr_newvars)

pretrained_train_op = pretrained_optimizer.minimize(loss, var_list=pretrained_vars, global_step=global_step)
newvars_train_op = newvars_optimizer.minimize(loss, var_list=new_vars)



with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # Create TF Coordinator
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    with coord.stop_on_exception():
        while not coord.should_stop():
            
            image_batch, label_batch = sess.run(batch)
            feed_dict = {
                x: image_batch,
                y: label_batch
            }

            fetches = [newvars_train_op, pretrained_train_op, loss, global_step]
            _, _, loss_val, i = sess.run(fetches, feed_dict=feed_dict)

            print("Step: %d" % i)

            if i % print_every == 0:
                acc_test = accuracy.eval(feed_dict={x: image_batch, y: label_batch})
                print(i, '\tloss = %0.4f' % loss_val + "\tTest accuracy: " + str(acc_test))

            if i % save_every == 0:
            	print('Saving checkpoint')
            	saver.save(sess, "/output/model" + str(i) + ".ckpt")

    saver.save(sess, "/output/model_final.ckpt")
    coord.join(threads)

    # for epoch in range(n_epochs):
    #     for iteration in range(train_dataset.num_examples // batch_size):
    #         X_batch, y_batch = train_dataset.next_batch(batch_size)
    #         sess.run(training_op, feed_dict={training: True, x: X_batch, y: y_batch})
        
    #     acc_test = accuracy.eval(feed_dict={x: test_dataset.images, y: test_dataset.labels})
    #     print(epoch, "Test accuracy:", acc_test)

    # save_path = saver.save(sess, "./lm_2class.ckpt")