'''
In this script, we provide a simple yet powerful function that uses 
image augmentation techniques.
This helps to deal with less number of training instances, 
increase accuracy, generalize models, etc.

Mandatory packages to be installed:
    tensorflow
    numpy    
'''

import tensorflow as tf
import numpy as np


def augment_data2(dataset, dataset_labels, label_counts):
	
	counts = label_counts.copy()
	augmented_images = []
	augmented_image_labels = []

	use_random_rotation=True
	use_random_shift=True   # This is no good ## Not enough RAM
	use_random_shear=True   # Not enough RAM  
	use_random_zoom=False
	num_augs_enabled = 0
	if use_random_rotation:
		num_augs_enabled = num_augs_enabled + 1
	if use_random_shift:
		num_augs_enabled = num_augs_enabled + 1
	if use_random_shear:
		num_augs_enabled = num_augs_enabled + 1
	if use_random_zoom:
		num_augs_enabled = num_augs_enabled + 1
	print("Num augs enabled: %d" % num_augs_enabled)
	aug_factors = dict()
	for ccval in range(0, 9):  # cloud coverage, values in [0,8]
		if num_augs_enabled == 0:
			continue
		aug_factors[ccval] = round((label_counts[8]/num_augs_enabled) / label_counts[ccval])

		"""
		print("dataset.load_training_data(): label %d, " 
			  "Aug_factor: %f, "
			  "Num images: %f, "
			  "Num images after oversampling: %f" %
			  (ccval,
			   aug_factors[ccval],
			   label_counts[ccval],
			   aug_factors[ccval] * label_counts[ccval] * num_augs_enabled))
		"""
	#maximg = 6000
	maximg = {0: 500, 1: 500, 2: 2000, 3: 2000, 4: 2000, 5: 2000, 6: 2000, 7: 500, 8: 100}
	imgcounts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
	for num in range (0, dataset.shape[0]):
		if num % 1000 == 0:
			print("Augmenting %d .." % num)
			
		cc = dataset_labels[num].tolist().index(1.0)
		if counts[cc] >= maximg[cc]:
				continue
		
		counts[cc] = counts[cc] + 1
				
		for i in range(0, 1) :
			
			
			if use_random_rotation is True:
				augmented_images.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset[num],
																							 45,
																							 row_axis=0,
																							 col_axis=1,
																							 channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])
				counts[cc] = counts[cc] + 1

			if  use_random_shear is True:
				augmented_images.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[num],
																						  0.45,
																						  row_axis=0,
																						  col_axis=1,
																						  channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])
				counts[cc] = counts[cc] + 1
				
			
			
			if   use_random_shift is True:
				augmented_images.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num],
																						  0.45,
																						  0.0,
																						  row_axis=0,
																						  col_axis=1,
																						  channel_axis=2,
																						  fill_mode='wrap'))
				augmented_image_labels.append(dataset_labels[num])
				counts[cc] = counts[cc] + 1
				

	print("Training imgcounts: ")
	print(imgcounts)
	print("Training images after augmentation: ")
	print(counts)
	return np.array(augmented_images), np.array(augmented_image_labels)
