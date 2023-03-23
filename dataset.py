

# This script was initially from a cv-tricks.com tutorial
# It has MIT licence


import re

import numpy as np
import os
from sklearn.utils import shuffle
import tensorflow as tf
import sys
import math

import matplotlib.pyplot as plt
import augment
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array

class DataSet(object):

    def __init__(self, images, labels):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels        
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return shuffle(self._images[start:end], self._labels[start:end])
        #return self._images[start:end], self._labels[start:end]


def load_training_data(labelsfile, imagedir, image_size, classes):
    images = []
    labels = []
    #img_names = []
    #cls = []


    label_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}

    count8 = 0
    
    print("Loading ..")
    with open(labelsfile, "r") as ins:
        for line in ins:

            myre = re.compile(r'(\S+)\s+(-?\d)$')
            mo = myre.search(line.strip())
            if mo is not None:
                path, cc = mo.groups()
            else:
                print("Error: No match")
                continue

            ## TODP: 
            if label_counts[int(cc)] >= 11000:
                continue

            try:
                image = load_img(imagedir + "/" + path)
                #img.show()
                #sys.exit(0)
            except Exception as e:
                print("load_img: %s %s " % imagedir + "/" + path,  str(e))
                continue

            if image is None:
                print("image %s is none" % path)
                continue
            # Already resized
            # image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            if int(cc) < 0:
                continue

            label_counts[int(cc)] = label_counts[int(cc)] + 1
            
            # opencv resize: 
            #image = image.astype(np.float32)
            # convert to numpy array
            img_array = img_to_array(image)
            # convert from [0:255] => [0.0:1.0]
            image = np.multiply(image, 1.0 / 255.0)
           
            images.append(image)
            labels.append(float(cc))

    images = np.array(images)
    labels = np.array(labels)

    print(label_counts)
    print(len(labels))
    return images, labels, label_counts

def read_train_sets2(labelsfile, imagedir, image_size, classes, validation_size):
   
    images, labels, label_counts = load_training_data(labelsfile, imagedir, image_size, classes)
    images, labels = shuffle(images, labels)
  
    print("SIZE: %d" % (sys.getsizeof(images) / (1024*1024)))
    
    #do_aug = True
    # Also augument test-set
    #if do_aug:
    #    print("Augmenting data ..")
    #    aug_images, aug_labels = augment.augment_data2(images, labels)
	
    #    images = np.concatenate([images, aug_images])
    #    labels = np.concatenate([labels, aug_labels])

    
    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    
    
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]

    print("train_images1: %d, train_labels1: %d" %(len(train_images), len(train_labels)))
    do_aug = True
    if do_aug:
        print("Augmenting data ..")
        aug_images, aug_labels = augment.augment_data2(train_images, train_labels)
        aug_images, aug_labels = shuffle(aug_images, aug_labels)

        train_images = np.concatenate([train_images, aug_images])
        train_labels = np.concatenate([train_labels, aug_labels])
        """
        plt.figure(figsize=(20,20))
        for i in range(0,50):
            plt.subplot(10,10,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            print(train_labels[i])
       
            plt.xlabel(train_labels[i])
            plt.imshow(train_images[i])
        plt.show()
        """
    
    return train_images, train_labels, validation_images, validation_labels
    

# Test
if __name__ == "__main__":
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    #load_training_data("alldata.txt", 128, classes)
    #load_training_data("alldata.txt", 128, classes)

    #data = read_train_sets("/home/espenm/data/v52/alldata.txt", "/home/espenm/data/v52/training_data", 128, classes, validation_size=0.35)
    train_images, train_labels, test_images, test_labels= read_train_sets2("/home/espenm/data/v52/alldata.txt", "/home/espenm/data/v52/training_data", 128, classes, validation_size=0.35)

    # To verify that the dataset looks correct, let's plot the first 25 images 
    # from the training set and display the class name below each image:
    """
    plt.figure(figsize=(5,5))
    for i in range(0,25):
        plt.subplot(10,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        print(train_labels[i])
        plt.xlabel(train_labels[i])
        plt.imshow(train_images[i])
  

    plt.show()
    """
