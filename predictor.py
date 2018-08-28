import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse


class Predictor(object):
    
	def __init__(self, trained_model_dir , checkpoint):
		#tf.logging.set_verbosity(tf.logging.ERROR)
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
		self.checkpoint = checkpoint
        
		#sys.stderr.write("Modeldata dircectory: %s\n" % trained_model_dir)
		#model_name =  os.path.basename(os.path.normpath(trained_model_dir))
		model_name =  "cc-predictor-model"
		sys.stderr.write("Model name: %s\n" % model_name)
		self.model_name = model_name
        
		self.checkpoint_file = trained_model_dir  + "/" + model_name 
		self.trained_model_dir = trained_model_dir
		self.meta = ""
		self.modelfile = "%s/%s-%d" % ( trained_model_dir, model_name, checkpoint)
		sys.stderr.write("modelfile: %s\n" % self.modelfile)                
		self.metafile = "%s-%d.meta" % (self.checkpoint_file, self.checkpoint)

        ## Let us restore the saved model -- OOBS: Doing this here gives different
        # results on the same image when predicting multiple times ! Move it to
        # The predict function
        #self.sess = tf.Session()
        # Step-1: Recreate the network graph. At this step only graph is
        # created.        
        #self.saver = tf.train.import_meta_graph(self.metafile)    
        # Step-2: Now let's load the weights saved using the restore method.        
        #self.saver.restore(self.sess, self.modelfile)
        # Accessing the default graph which we have restored
        #self.graph = tf.get_default_graph()       
        
	def predict(self, image_path):
        
		image_size=128
		num_channels=3
		images = []
        
		# Resizing the image to our desired size and preprocessing will be
		# done exactly as done during training
		try:
			# Reading the image using OpenCV
			image = cv2.imread(image_path)
			image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
		except cv2.error as e:
			sys.stderr.write(str(e) + "\n")
			sys.stderr.write("Corrupt image. Skipping image_path. %s\n" % image_path)
			return -1
		images.append(image)
		images = np.array(images, dtype=np.uint8)

		# Convert from [0, 255] -> [0.0, 1.0].
		images = images.astype('float32')
		images = np.multiply(images, 1.0/255.0) 
        
		## Let us restore the saved model 
		self.sess = tf.Session()
		# Step-1: Recreate the network graph. At this step only graph is
		# created.        
		#saver = tf.train.import_meta_graph(self.metafile)    
		# Step-2: Now let's load the weights saved using the restore method.        
		#saver.restore(self.sess, self.modelfile)
		# Accessing the default graph which we have restored
		self.graph = tf.get_default_graph()


		tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], "cc-predictor-model")

		
		# Now, let's get hold of the op that we can be processed to get the
		# output.
		# In the original network y_pred is the tensor that is the prediction
		# of the network
		self.y_pred = self.graph.get_tensor_by_name("y_pred:0")
    
		## Let's feed the images to the input placeholders
		self.x = self.graph.get_tensor_by_name("x:0") 
		self.y_true = self.graph.get_tensor_by_name("y_true:0") 
		self.y_test_images = np.zeros((1, 9)) 
        
		# The input to the network is of shape
		# [None image_size image_size num_channels].
		# Hence we reshape.
		x_batch = images.reshape(1, image_size,image_size,num_channels)
                

		## Creating the feed_dict that is required to be fed to calculate
		# y_pred 
		feed_dict_testing = {self.x: x_batch, self.y_true: self.y_test_images}
		result=self.sess.run(self.y_pred, feed_dict=feed_dict_testing)
		self.sess.close()
		self.sess = None
		#print(result)
		return result


    
