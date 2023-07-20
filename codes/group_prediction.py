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
# ------------------------------------------------------------------------------------------------
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
# print(pos_examples)
# print(neg_examples)
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
        test_image = image.load_img(img_path, target_size=(SIZE, SIZE))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image / 255.0)
        # print(result)
        #training_set.class_indices
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

print(f'TP: {TP} TN: {TN} FN: {FN} FP: {FP}')

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

ax.set_xlim([0.5, 1])
ax.set_ylim([0, 1])

fsize = 20
ax.set_xlabel('Recall', fontsize = fsize)
ax.set_ylabel('Precision', fontsize = fsize)
ax.set_title('Precision-Recall Performance', fontsize = fsize)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(fontsize = fsize-4)
#plt.show()
plt.savefig('output/precision_recall_group_prediction.jpg', dpi = 300)

