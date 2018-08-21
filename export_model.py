#!/usr/bin/python3 -u

import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import math
import predictor
from tensorflow.python.framework import graph_util

# Export a model for use with  tensorflow_model_server or
# for use with for instance a go-program .

cpdir = './modeldata'

checkpoint = 104


model_name =  "cc-predictor-model"

checkpoint_file = cpdir  + "/" + model_name
modelfile = "%s/%s-%d" % ( cpdir, model_name, checkpoint)
metafile = "%s-%d.meta" % (checkpoint_file, checkpoint)

sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is
# created.        
saver = tf.train.import_meta_graph(metafile)    
# Step-2: Now let's load the weights saved using the restore method.        
saver.restore(sess, modelfile)
# Accessing the default graph which we have restored
graph = tf.get_default_graph()

input_graph_def = graph.as_graph_def()



#builder = tf.saved_model.builder.SavedModelBuilder("cc-predictor-model")

y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")

tf.saved_model.simple_save(sess, "cc-predictor-model-s",
						   inputs={"x": x, "y_true": y_true},
						   outputs={"y_pred", y_pred})

#tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
#tensor_info_y = tf.saved_model.utils.build_tensor_info(y_pred)
#y_test_images = np.zeros((1, 9))

"""
prediction_signature = (
	tf.saved_model.signature_def_utils.build_signature_def(
		inputs={'input': tensor_info_x},
		outputs={'output': tensor_info_y},
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

