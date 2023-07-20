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
# Deploy ML model on GPUs using mirrored strategy
# tensorflow-gpu should be correctly updated, otherwise the strategy will not work
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# The code below may have to be adjusted in different machines.
mirrored_strategy = tf.distribute.MirroredStrategy(devices= ["/gpu:0","/gpu:1"],
                                                   cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with mirrored_strategy.scope():
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
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#------------------------------------------------------------------------------------------------

'''
# Save CNN model
# Save the entire model to a HDF5 file.
cnn.save('cnn.h5')

# Recreate the exact same model, including its weights and the optimizer
# new_model = tf.keras.models.load_model('cnn.h5')
# Show the model architecture
# new_model.summary()

# Save only weights
cnn.save_weights('cnn_weights.h5')

cnn.load_weights('cnn_weights.h5')

# Save only architecture
jason_string = cnn.to_json()
with open("cnn_model.json", "w") as f:
    f.write(jason_string)

with open("cnn_model.json", "r") as f:
    loaded_json_string = f.read()
    
#new_model = keras.model.model_from_json(loaded_json_string)
#print(new_model.summary())
'''

#------------------------------------------------------------------------------------------------
# Making a single prediction
'''
def single_predict(fileName):
    path = 'iccad1/single_prediction/' + fileName + '.png'
    test_image = image.load_img(path, target_size = (SIZE, SIZE))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    #print(test_image)
    result = cnn.predict(test_image/255.0)
    print(result)
    training_set.class_indices
    print(training_set.class_indices)
    if result[0][0] > 0.5:
      prediction = 'HS'
    else:
      prediction = 'NHS'

    print(prediction)
    
def showPNG(fileName):
    path = 'iccad1/single_prediction/' + fileName + '.png'
    display(Image(path, width = 200, height = 200))

showPNG('test_HS12')
single_predict('test_HS12')
showPNG('test_NHS133.png1')
single_predict('test_NHS133.png1')
'''


#------------------------------------------------------------------------------------------------
# Making group predictions¶
path = 'group_prediction/'
files = os.listdir(path)
pos_examples = []
neg_examples = []
for fileName in files:
    if fileName.startswith("HS"):
        pos_examples.append((fileName, 1))
    else:
        neg_examples.append((fileName, 0))

test_img_count = len(files)        
pos_count = len(pos_examples)
neg_count = len(neg_examples)
print(test_img_count, "images are loaded, including", pos_count, "HS patterns and", neg_count, "NHS patterns.")
#print(pos_examples)
#print(neg_examples)
labeled_files = copy.deepcopy(pos_examples)
labeled_files.extend(neg_examples)
print(labeled_files)

# Loop through images and calculate confusion matrix
def group_predict(labeled_files, path):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for labeled_file in labeled_files:
        img_path = path + labeled_file[0]
        test_image = image.load_img(img_path, target_size = (SIZE, SIZE))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image/255.0)
        #print(result)
        training_set.class_indices
        if result[0][0] > 0.5:
          prediction = 1
        else:
          prediction = 0
        
        if labeled_file[1] == 1 and prediction == 1:
            TP += 1
        elif labeled_file[1] == 0 and prediction == 0:
            TN += 1
        elif labeled_file[1] == 1 and prediction == 0:
            FN += 1
        elif labeled_file[1] == 0 and prediction == 1:
            FP += 1
        
    return (TP, TN, FP, FN)
    
(TP, TN, FP, FN) = group_predict(labeled_files, path)

# Calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1_score = 2 * precision * recall / (precision + recall)
print(TP, TN, FP, FN)
print(accuracy, recall, precision, F1_score)

# Plot precision-recall performance and compare with other researchers' work
# Values from previous researchers 
# Shin, M.; Lee, J.-H. Accurate Lithography Hotspot Detection Using Deep Convolutional Neural Networks. J. Micro/Nanolith.
# MEMS MOEMS 2016, 15, 043507.
# Zhou, K.; Zhang, K.; Liu, J.; Liu, Y.; Liu, S.; Cao, G.; Zhu, J. An Imbalance Aware Lithography Hotspot Detection Method Based
# on HDAM and Pre-trained GoogLeNet. Meas. Sci. Technol. 2021, 32, 125008.
# Liao, L.; Li, S.; Che, Y.; Shi, W.;Wang, X. Lithography Hotspot Detection Method Based on Transfer Learning Using Pre-Trained
# Deep Convolutional Neural Network. Appl. Sci. 2022, 12, 2192. https://doi.org/10.3390/app12042192

Shin_recall = [0.951, 0.988, 0.975, 0.938, 0.927]
Shin_precision = [0.358, 0.216, 0.199, 0.157, 0.181]
Zhou_recall = [0.995, 0.986, 0.982, 0.972, 0.980]
Zhou_precision = [0.324, 0.702, 0.443, 0.355, 0.549]
Liao_recall = [0.947, 0.988, 0.988, 0.937, 1.000]
Liao_precision = [0.968, 0.850, 0.967, 0.913, 0.737]

dia = 100
colors = ['black','green','blue','red']
markers = ['o', '^', 's', 'P']
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 6))
ax.scatter(Shin_recall, Shin_precision, label='Shin (2016)', c = colors[0], s = dia, marker = markers[0])
ax.scatter(Zhou_recall, Zhou_precision, label='Zhou (2021)', c = colors[1], s = dia, marker = markers[1])
ax.scatter(Liao_recall, Liao_precision, label='Liao (2022)', c = colors[2], s = dia, marker = markers[2])
ax.scatter(recall, precision, label='Our Model', c = colors[3], s = 4 * dia, marker = markers[3])

ax.set_xlim([0.9, 1])
ax.set_ylim([0, 1])

fsize = 20
ax.set_xlabel('Recall', fontsize = fsize)
ax.set_ylabel('Precision', fontsize = fsize)
ax.set_title('Precision-Recall Performance', fontsize = fsize)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(fontsize = fsize-4)
plt.show()

