#!/usr/bin/python3 -u

import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import math
import predictor

# Export a model for use with  tensorflow_model_server or
# for use with for instance a go-program .

cpdir = './modeldata'

# v11-python3-rotate-augmentation .. 
checkpoint = 347


model_name =  "cc-predictor-model"

checkpoint_file = cpdir  + "/" + model_name
modelfile = "%s/%s-%d" % ( cpdir, model_name, checkpoint)
metafile = "%s-%d.meta" % (checkpoint_file, checkpoint)

sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is
# created.        
saver = tf.train.import_meta_graph(metafile)

saver.restore(sess, modelfile)
builder = tf.saved_model.builder.SavedModelBuilder("cc-predictor-model")

# GOLANG note that we must tag our model so that we can retrieve it at inference-time
builder.add_meta_graph_and_variables(sess,["serve"])

builder.save()

