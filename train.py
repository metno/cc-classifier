#!/usr/bin/python3 -u

import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import datetime
import re
from numpy.random import seed
from tensorflow import set_random_seed
import argparse
import os

# This script was initially from a cv-tricks.com tutorial
# It has a MIT licence

# Hyper params

BATCH_SIZE        = 12
DROPOUT_KEEP_PROB = 0.5
LEARNING_RATE     = 1e-7
# Train/validation split 30% of the data will automatically be used for validation
VALIDATION_SIZE = 0.30
use_L2_Regularization = True
# L2 regularization. This is a good beta value to start with ? 
BETA = 0.01

parser = argparse.ArgumentParser(description='Train a cnn for predicting cloud coverage')
parser.add_argument('--labelsfile', type=str, help='A labels file containing lines like this: fileNNN.jpg 6')
parser.add_argument('--imagedir', type=str, help='The training and validation data')
parser.add_argument('--outputdir', type=str, default='modeldata', help='where to write model snapshots')
parser.add_argument('--inputdir', type=str, default=None, help='Start training on exising model')

parser.add_argument('--epoch', type=str, default=None, help='Start training from epoch')


parser.add_argument('--logdir', type=str, default='/tmp/tf', help='Metrics data')
args = parser.parse_args()

logs_path = args.logdir

# For tensorboard
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):

    # Define the weights that will be trained.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    #variable_summaries(weights)

    ## Create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)
    #variable_summaries(biases)
    
    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer




def create_flatten_layer(layer):
    # We know that the shape of the layer will be [BATCH_SIZE img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    weights,
                    biases,
                    use_relu=True):

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices,
    # we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer, name='activation')

    tf.summary.histogram('activations', layer)

    return layer


def show_progress(iteration, epoch, feed_dict_train, feed_dict_validate, tr_acc):
    #acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    val_loss = session.run(cost, feed_dict=feed_dict_validate)

    msg = "Iteration {4} Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print("%s %s" % (msg.format(epoch + 1, tr_acc, val_acc, val_loss, iteration +1), datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))


def train(start, num_iterations):

    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    for i in range(start, num_iterations):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        x_valid_batch, y_valid_batch = data.valid.next_batch(BATCH_SIZE)


        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch,
                        keep_prob: DROPOUT_KEEP_PROB
        }
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}


        summary, _, tr_acc = session.run([merged, optimizer, accuracy],
                                 feed_dict_tr)
       
    
        
        if i % int(data.train.num_examples/BATCH_SIZE) == 0:
            # For tensorboard:
            # train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            
            summary, acc_v = session.run([merged, accuracy], feed_dict=feed_dict_val)
            # Tensorboard:
            test_writer.add_summary(summary, i)
            test_writer.flush()
            train_writer.flush()        
    
            epoch = int(i / int(data.train.num_examples/BATCH_SIZE))
            
            show_progress(i, epoch, feed_dict_tr, feed_dict_val, tr_acc)

            saver.save(session, args.outputdir + '/cc-predictor-model', global_step=epoch)

            # Export the model for use with other languages
            """
            builder = tf.saved_model.builder.SavedModelBuilder("cc-predictor-model-%d" % epoch)
            tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(y_pred)

            prediction_signature = (
                    tf.saved_model.signature_def_utils.build_signature_def(
                            inputs={'input': tensor_info_x},
                            outputs={'output': tensor_info_y},
                            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))



            builder.add_meta_graph_and_variables(
                    session, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={
                            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            prediction_signature,
                    },
            )

            builder.save(as_text=False)
            """
            #tf.saved_model.simple_save(session, "cc-predictor-model-%d" % i, inputs=feed_dict_tr, outputs=feed_dict_val)

if __name__ == "__main__":

    if args.epoch is not None: # If set we continue training from where we left
        os.system("rm -rf /tmp/tf")
    retval = os.system("mkdir -p " + args.outputdir)
    if retval != 0:
        sys.stderr.write('Could not create outputdir\n')
        sys.exit(63)

    print("BATCH_SIZE: %d" % BATCH_SIZE)
    print("DROPOUT_KEEP_PROB %f" % DROPOUT_KEEP_PROB)
    print("LEARNING_RATE: %f"% LEARNING_RATE)
    # Train/validation split 30% of the data will automatically be used for validation
    print("VALIDATION_SIZE: %f" %  VALIDATION_SIZE)

        
    #Adding Seed so that random initialization is consistent
    seed(1)
    set_random_seed(2)

    #Prepare input data
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    num_classes = len(classes)


    img_size = 128
    num_channels = 3

    # We shall load all the training and validation images and labels into memory
    # using openCV and use that during training
    data = dataset.read_train_sets(args.labelsfile, args.imagedir, img_size, classes, validation_size=VALIDATION_SIZE)

    print("Complete reading input data. ")
    print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
    print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))
    print("data.train.num_examples: %d" % data.train.num_examples)

    session = tf.Session()

    # GOLANG note that we must label the input-tensor! (name='x')
    x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

    ## labels
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)    

    layer_conv1 = create_convolutional_layer(input=x,
                                             num_input_channels=3,
                                             conv_filter_size=128,
                                             num_filters=3,
    )
    layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                             num_input_channels=3,
                                             conv_filter_size=64,
                                             num_filters=3,
    )

    layer_conv3= create_convolutional_layer(input=layer_conv2,
                                        num_input_channels=3,
                                        conv_filter_size=32,
                                        num_filters=3,
    )

    layer_conv4= create_convolutional_layer(input=layer_conv3,
                                        num_input_channels=3,
                                        conv_filter_size=16,
                                        num_filters=3,
    )

    layer_conv5= create_convolutional_layer(input=layer_conv4,
                                        num_input_channels=3,
                                        conv_filter_size=8,
                                        num_filters=3,
    )


    layer_flat = create_flatten_layer(layer_conv5)

    #Let's define trainable weights and biases for the fully connected layer1.
    num_inputs=layer_flat.get_shape()[1:4].num_elements()
    num_outputs=128
    fc1_weights = create_weights(shape=[num_inputs, num_outputs])
    fc1_biases = create_biases(num_outputs)    
    layer_fc1 = create_fc_layer(input=layer_flat,                                
                                weights=fc1_weights,
                                biases=fc1_biases,
                                use_relu=True
    )

    # Remember: Dropout should only be introduced during training, not evaluation,
    # otherwise your evaluation results would be stochastic as well. 
    # Argument to droupout is the probability of _keeping_ the neuron:
    keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
    dropped = tf.nn.dropout(layer_fc1, keep_prob)
    num_inputs=128
    num_outputs=num_classes
    fc2_weights = create_weights(shape=[num_inputs, num_outputs])
    fc2_biases = create_biases(num_outputs)


    layer_fc2 = create_fc_layer(input=dropped,                                
                                weights=fc2_weights,
                                biases=fc2_biases,
                                use_relu=False
    )

    # Softmax is a function that maps [-inf, +inf] to [0, 1] similar as Sigmoid. But Softmax also
    # normalizes the sum of the values(output vector) to be 1.
    y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
    

    
    

    # Logit is a function that maps probabilities [0, 1] to [-inf, +inf].
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                               labels=y_true)

    # validation_cost = tf.reduce_mean(cross_entropy)
    
    
    # cost = loss
    if use_L2_Regularization: # Loss function using L2 Regularization                 
        regularizer = tf.nn.l2_loss(fc2_weights)
        cost = tf.reduce_mean(cross_entropy + BETA * regularizer)
    else:
        cost = tf.reduce_mean(cross_entropy)
        
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Note that we must label the infer-operation for use from go!!
    y_pred_cls = tf.argmax(y_pred, axis=1, name="infer")
    # This converge fast and should be good enough for our use. Lets use this.
    # turning it off for testing :
    #correct_prediction = tf.abs(tf.subtract(y_pred_cls, y_true_cls)) <= 1
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a summary to monitor cost tensor
    tf.summary.scalar("Loss", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("Accuracy", accuracy)

    #tf.summary.scalar('cross_entropy', cross_entropy)

    # merge all summaries into a single "operation" which we can execute in a session
    merged = tf.summary.merge_all()
    # create log writer object

    train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
    test_writer  = tf.summary.FileWriter(logs_path + '/test')

    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=100000)
    path = args.inputdir
    start = 0

    #if path is not None and tf.train.latest_checkpoint(path) is not None:
    if path is not None and args.epoch is not None:
        print("Loading %s  %s " % (path, path + "/cc-predictor-model-" + args.epoch))
        print("Try restoring model ..")
        saver.restore(session, path + "/cc-predictor-model-" + args.epoch)        
        print("Training from epoch %d" % int(args.epoch))
        start = int(args.epoch)  * int(data.train.num_examples/BATCH_SIZE) + 2
        print("StartIter: %d " % start)
    train(start, num_iterations=100000000)
