# CNN Model for Hotspot Detection on ICCAD2012 Dataset
# Author: Michael Qu
# Creation Date: 8/25/2022

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from IPython.display import Image
import time
import os
import copy

# Global Parameters
SIZE = 128
# Image Parameters in the future can be added to the channels
CHANNELS = 3
DROPOUT_RATE = 0.5
BATCH_SIZE = 32
CLASSES = {'NHS': 0, 'HS': 1}


# Recreate the exact same model, including its weights and the optimizer
cnn = tf.keras.models.load_model('cnn.h5')
# Show the model architecture
cnn.summary()

def single_predict(fileName):
    path = 'group_prediction/' + fileName + '.png'
    test_image = image.load_img(path, target_size = (SIZE, SIZE))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    #print(test_image)
    result = cnn.predict(test_image/255.0)
    print(result)
    #training_set.class_indices
    #print(training_set.class_indices)
    if result[0][0] > 0.5:
      prediction = 'HS'
    else:
      prediction = 'NHS'

    print(prediction)

single_predict('HS0')
single_predict('HS10')
single_predict('NHS0.png1')
single_predict('NHS0.png2')
