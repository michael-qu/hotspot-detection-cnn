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
CLASSES = {'NHS':0, 'HS':1}

#------------------------------------------------------------------------------------------------
# Data Preprocessing
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip = True,
                                   vertical_flip = True)
training_set = train_datagen.flow_from_directory(directory = 'iccad1/train',
                                                 target_size = (SIZE, SIZE),
                                                 batch_size = BATCH_SIZE,
                                                 classes = CLASSES,
                                                 class_mode = 'binary',
                                                 shuffle = True,
                                                 seed = 42
                                                )

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(directory = 'iccad1/test',
                                            target_size = (SIZE, SIZE),
                                            batch_size = BATCH_SIZE,
                                            classes = CLASSES,
                                            class_mode = 'binary',
                                            shuffle = True,
                                            seed = 42
                                            )
#------------------------------------------------------------------------------------------------
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Convolutional & Pooling Layers
cnn.add(tf.keras.layers.Conv2D(filters=4, kernel_size=3, activation='relu', input_shape=[SIZE, SIZE, CHANNELS]))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full Connection Layers
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
cnn.add(tf.keras.layers.Dropout(DROPOUT_RATE))
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(tf.keras.layers.Dropout(DROPOUT_RATE))

# Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# For multiclass classification, softmax should be used as activation function.
#cnn.add(tf.keras.layers.Dense(units=n, activation='softmax'))

cnn.summary()

#------------------------------------------------------------------------------------------------
# Compile CNN
opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.99)
# opt = 'adam'  #This optimizer can be used for comparison
cnn.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
# For multiclass classification:
# cnn.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Training the CNN on the training set and evaluating it on the test set
start_time = time.time()
history = cnn.fit(x = training_set, validation_data = test_set, epochs = 16)
end_time = time.time()
print("\n********Time used for training: ", end_time - start_time)

# Plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show()
plt.savefig('output/loss_curve.jpg', dpi = 300)
plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#plt.show()
plt.savefig('output/accuracy_curve.jpg', dpi = 300)
plt.clf()

#------------------------------------------------------------------------------------------------

# Save CNN model
# Save the entire model to a HDF5 file.
cnn.save('cnn.h5')