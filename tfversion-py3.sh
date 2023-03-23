#!/usr/bin/env python3

#python3 -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 3

import tensorflow as tf
print(tf.__version__)
tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
