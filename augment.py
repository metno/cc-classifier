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
import math

def flip_axis(x, axis):
    cp = np.copy(x)
    cp = np.asarray(cp).swapaxes(axis, 0)
    cp = cp[::-1, ...]
    cp = cp.swapaxes(0, axis)
    return cp


def salt_and_pepper_noise(image):
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.00008
    out = np.copy(image)
    # Salt mode
    #num_salt = np.ceil(amount * image.size * s_vs_p)
    #coords = [np.random.randint(0, i - 1, int(num_salt))
    #          for i in image.shape]
    #out[tuple(coords)] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[tuple(coords)] = 0
    return out

def augment_data2(dataset, dataset_labels, label_counts):

    counts = label_counts.copy()
    print("COunts: ")
    
    augmented_images = []
    augmented_image_labels = []

    use_flip_axis             = True
    use_random_rotation       = True
    use_random_shift          = False
    use_random_shear          = False
    use_copy                  = False
    use_salt_and_pepper_noise = False

    num_augs_enabled = 0

    if use_salt_and_pepper_noise:
        num_augs_enabled = num_augs_enabled + 1
    if use_flip_axis:
        num_augs_enabled = num_augs_enabled + 1
    if use_copy:
        num_augs_enabled = num_augs_enabled + 1
    if use_random_rotation:
        num_augs_enabled = num_augs_enabled + 1
    if use_random_shift:
        num_augs_enabled = num_augs_enabled + 1
    if use_random_shear:
        num_augs_enabled = num_augs_enabled + 1

    print("Num augs enabled: %d" % num_augs_enabled)
    aug_factors = dict()
    for ccval in range(0, 9):  # cloud coverage, values in [0,8]
        if num_augs_enabled == 0:
            continue
        aug_factors[ccval] = math.ceil((float(label_counts[8])/float(num_augs_enabled)) / float(label_counts[ccval]))
    print(aug_factors)
        
    maximg = {0: 5000, 1: 5000, 2: 5000, 3: 5000, 4: 5000, 5: 5000, 6: 5000, 7: 5000, 8: 5000}

    for num in range (0, dataset.shape[0]):
        if num % 1000 == 0:
            print("Augmenting %d .." % num)

        cc = dataset_labels[num].tolist().index(1.0)
        if counts[cc] >= maximg[cc]:
            continue

        for i in range(0, aug_factors[cc] + 2) :

            if use_flip_axis is True:
                augmented_images.append(flip_axis(dataset[num],1))
                augmented_image_labels.append(dataset_labels[num])
                counts[cc] = counts[cc] + 1

            if use_salt_and_pepper_noise is True:
                augmented_images.append(salt_and_pepper_noise(dataset[num]))
                augmented_image_labels.append(dataset_labels[num])
                counts[cc] = counts[cc] + 1

            if use_copy is True:
                augmented_images.append(dataset[num].copy())
                augmented_image_labels.append(dataset_labels[num])
                counts[cc] = counts[cc] + 1

            if use_random_rotation is True:
                augmented_images.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset[num],
                                                                                             1.0,
                                                                                             row_axis=0,
                                                                                             col_axis=1,
                                                                                             fill_mode='reflect',
                                                                                             channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])
                counts[cc] = counts[cc] + 1

            if use_random_shear is True:
                augmented_images.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[num],
                                                                                          1.0,
                                                                                          row_axis=0,
                                                                                          col_axis=1,
                                                                                          channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])
                counts[cc] = counts[cc] + 1



            if use_random_shift is True:
                augmented_images.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num],
                                                                                          0.01,
                                                                                          0.0,
                                                                                          row_axis=0,
                                                                                          col_axis=1,
                                                                                          channel_axis=2,
                                                                                          fill_mode='wrap'))
                augmented_image_labels.append(dataset_labels[num])
                counts[cc] = counts[cc] + 1


    print("Training images after augmentation: ")
    print(counts)
    return np.array(augmented_images), np.array(augmented_image_labels)
