#!/bin/sh
''''exec python3 -u -- "$0" ${1+"$@"} # '''
# vi: syntax=python


import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import math
import predictor
from tensorflow.python.framework import graph_util
import argparse

# Export a model for use with  tensorflow_model_server or
# for use with for instance a go-program .

parser = argparse.ArgumentParser(description='Exports a model tensorflow model for use by external programs')
parser.add_argument('--modeldir', type=str, default=None, help='modeldir')
parser.add_argument('--epoch', type=str, default=None, help='Epoch/checkpoint to load')
args = parser.parse_args()

cpdir = args.modeldir
checkpoint = int(args.epoch)


model_name =  "cc-predictor-model"

checkpoint_file = cpdir  + "/" + model_name
modelfile = "%s/%s-%d" % ( cpdir, model_name, checkpoint)
metafile = "%s-%d.meta" % (checkpoint_file, checkpoint)

sess = tf.Session()
sess.run([tf.local_variables_initializer(), tf.tables_initializer()])
#sess.run(tf.initialize_all_variables())
#sess.run(tf.initialize_local_variables())


# Step-1: Recreate the network graph. At this step only graph is
# created.        
saver = tf.train.import_meta_graph(metafile)    
# Step-2: Now let's load the weights saved using the restore method.        
saver.restore(sess, modelfile)
# Accessing the default graph which we have restored
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
#y_pred_cls = graph.get_tensor_by_name("infer:0")
y_pred_cls = tf.argmax(y_pred, axis=1, name="infer")

keep_prob = graph.get_tensor_by_name("keep_prob:0")

is_training = graph.get_tensor_by_name("is_training:0")

tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
tensor_info_y_pred = tf.saved_model.utils.build_tensor_info(y_pred)
tensor_info_y_true = tf.saved_model.utils.build_tensor_info(y_true)
tensor_info_y_pred_cls = tf.saved_model.utils.build_tensor_info(y_pred_cls)
y_test_images = np.zeros((1, 9)) 

keep_place =  tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
istrain_place = tf.placeholder_with_default(False, shape=(), name="is_training")

tf.saved_model.simple_save(sess,
            "cc-predictor-model",
            inputs={"x": x, "y_true": y_true},
            #outputs={"infer": y_pred_cls, "keep_prob": keep_place, "is_training": istrain_place})
	    outputs={"infer": y_pred_cls})


#builder = tf.saved_model.builder.SavedModelBuilder('cc-predictor-model')
#builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
#builder.save()


"""
prediction_signature = (
	tf.saved_model.signature_def_utils.build_signature_def(
		inputs={'input': tensor_info_x},
		outputs={'predict': tensor_info_y_true},
		method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
 
			
 
builder.add_meta_graph_and_variables(
	sess, [tf.saved_model.tag_constants.SERVING],
	signature_def_map={
		tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
		prediction_signature,
	},
)

builder.save()
"""


						


