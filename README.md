# skin cancer detection
## Introduction
This code performs skin data analysis using TensorFlow and Keras. It trains a deep learning model to classify skin images into different categories. The code utilizes various preprocessing techniques and data augmentation methods to improve the model's performance.

## Prerequisites
Make sure you have the following libraries installed:
- pathlib
- tensorflow
- matplotlib
- numpy
- pandas
- PIL
- keras
- seaborn
- glob

## Code
```python
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
import random
from glob import glob
import seaborn as sns
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import matplotlib.image as img
import warnings

# Set up environment
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

# Set memory growth
try:
    tf.config.experimental.set_memory_growth = True
except Exception as ex:
    print(e)

# Define data directories
data_dir_train = pathlib.Path("/content/drive/MyDrive/skin_data/train")
data_dir_test = pathlib.Path("/content/drive/MyDrive/skin_data/test")

# Count images in training and testing sets
image_count_train = len(list(data_dir_train.glob('*.jpg')))
print(image_count_train)

image_count_test = len(list(data_dir_test.glob('*.jpg')))
print(image_count_test)

# Prepare data
batch_size = 32
img_height = 180
img_width = 180
rnd_seed = 123
random.seed(rnd_seed)

# Load training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Load validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Load testing dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_test,
  validation_split=0.9,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Get class names
class_names = train_ds.class_names
print(class_names)

# Display sample images
num_classes = len(class_names)
plt.figure(figsize=(10,10))
for i in range(num_classes):
  plt.subplot(3,3,i+1)
  image = img.imread(str(list(data_dir_train.glob(class_names[i]+'/*.jpg'))[1]))
  plt.title(class_names[i])
  plt.imshow(image)

# Check batch shapes
for image_batch, labels_batch in train_ds.take(1):
    print(image_batch.shape)
    print(labels_batch.shape)

# Configure dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().pref
