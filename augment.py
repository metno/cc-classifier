


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


'''
    This function peforms various data augmentation techniques to the dataset
    
    @parameters:
        dataset: the feature training dataset in numpy array with 
        shape [num_examples, num_rows, num_cols, num_channels] (since it is an image in numpy array)
        dataset_labels: the corresponding training labels of the feature training dataset in the 
        same order, and numpy array with shape [num_examples, <anything>]
        augmentation_factor: how many times to perform augmentation.
        use_random_rotation: whether to use random rotation. default: true
        use_random_shift: whether to use random shift. default: true
        use_random_shear: whether to use random shear. default: true
        use_random_zoom: whether to use random zoom. default: true
        skip_labels: Donty augument images in this list
    @returns:
        augmented_images: augmented dataset
        augmented_image_labels: labels corresponding to augmented dataset in order.
        
    for the augmentation techniques documentation, go here:
    	https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_rotation
    	https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_shear
        https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_shift
        https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_zoom
'''
def augment_data(dataset, dataset_labels,
				 aug_factors, 
				 use_random_rotation=True,
                 use_random_shear=True,
                 use_random_shift=True,
                 use_random_zoom=True,
				 
                 skip_labels = [] ):	

	augmented_images = []
	augmented_image_labels = []

	count = 0
	for num in range (0, dataset.shape[0]):
		if count % 1000 == 0:
			print("Augmenting %d" % count)
		count = count + 1
		#print(dataset_labels[num])
		skip = False
		for l in range (0, len(skip_labels)):
			if dataset_labels[num][skip_labels[l]] == 1:
				#print("Skipping augmentation for label %d" % skip_labels[l])
				skip = True
				break
		if skip:
			continue

		i = dataset_labels[num].tolist().index(1.0)
		#print(i)
		augementation_factor = aug_factors[i]
		for i in range(0, augementation_factor):
			# original image:
			#augmented_images.append(dataset[num])
			#augmented_image_labels.append(dataset_labels[num])

			if use_random_rotation:
				augmented_images.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset[num],
															20,
															row_axis=0,
															col_axis=1,
															channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_shear:
				augmented_images.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[num],
															   0.2,
															   row_axis=0,
															   col_axis=1,
															   channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_shift:
				augmented_images.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num],
														 0.2,
														 0.2,
														 row_axis=0,
														 col_axis=1,
														 channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_zoom:
				augmented_images.append(tf.contrib.keras.preprocessing.image.random_zoom(dataset[num],
														[0.2, 0.1],
														row_axis=0,
														col_axis=1,
														channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

	return np.array(augmented_images), np.array(augmented_image_labels)


def augment_data2(dataset, dataset_labels, big, label_counts, aug_factors):
	
	counts = label_counts.copy()
	augmented_images = []
	augmented_image_labels = []
	
	for num in range (0, dataset.shape[0]):
		if num % 1000 == 0:
			print("Augmenting %d .." % num)
			
		cc = dataset_labels[num].tolist().index(1.0)
		augementation_factor = aug_factors[cc]
		for i in range(0, augementation_factor):
			if counts[cc] < counts[big]:
				augmented_images.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset[num],
																							 20,
																							 row_axis=0,
																							 col_axis=1,
																							 channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])
				counts[cc] = counts[cc] + 1

			if counts[cc] < counts[big]:
				augmented_images.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[num],
																						  0.2,
																						  row_axis=0,
																						  col_axis=1,
																						  channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])
				counts[cc] = counts[cc] + 1
				
			
			"""	
			if counts[cc] < counts[big]:
				augmented_images.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num],
																						  0.2,
																						  0.2,
																						  row_axis=0,
																						  col_axis=1,
																						  channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])
				counts[cc] = counts[cc] + 1
			"""	
		#print("Num %d's: %d" %  (cc, counts[cc]))
	print(counts)
	return np.array(augmented_images), np.array(augmented_image_labels)
