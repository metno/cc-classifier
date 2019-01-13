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
    augmented_images = []
    augmented_image_labels = []

    use_flip_axis             = True
    use_random_rotation       = False
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
        aug_factors[ccval] = round((label_counts[8]/num_augs_enabled) / label_counts[ccval])
    print(aug_factors)
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

    #maximg = {0: 3000, 1: 3000, 2: 3000, 3: 3000, 4: 3000, 5: 3000, 6: 3000, 7: 0, 8: 0}

    # v28 # localhost


    # Enough RAM on Floydhub
    #maximg = {0: 8000, 1: 8000, 2: 8000, 3: 8000, 4: 12000, 5: 12000, 6: 12000, 7: 12000, 8: 0}
    #maximg = {0: 9000, 1: 9000, 2: 9000, 3: 9000, 4: 9000, 5: 9000, 6: 9000, 7: 9000, 8: 0}
    #maximg = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    # Enough RAM on neo
    maximg = {0: 8000, 1: 8000, 2: 8000, 3: 8000, 4: 8000, 5: 8000, 6: 8000, 7: 8000, 8: 8000}
    for num in range (0, dataset.shape[0]):
        if num % 1000 == 0:
            print("Augmenting %d .." % num)

        cc = dataset_labels[num].tolist().index(1.0)
        if counts[cc] >= maximg[cc]:
            continue

        for i in range(0, aug_factors[cc] + 1) :

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
