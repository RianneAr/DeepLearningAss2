#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import keras
# import keras_tuner
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras import backend as K
# from keras.engine import training
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from functools import partial
import tensorflow.experimental.numpy as tnp


# ## Loading data
# 
# First, we load in the data and split it into a train and test set. 

# In[2]:


# load the grayscale images and two-integer labels
images = np.load('images.npy')
labels = np.load('labels.npy')
images = images.astype('float32')
images /= 255

# input image dimensions
img_rows, img_cols = 150, 150

# reshape the 2D input data to 4D to fit the conv2D layer
if K.image_data_format() == 'channels_first':
    images = images.reshape(images.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    images = images.reshape(images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# the data, split between train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[3]:


plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i],cmap='gray')
plt.show()


# ## Regression

# In[4]:


# basic parameter settings
batch_size = 32 # 32>64>128
epochs = 160

# change the two-interger labels to a float
y_reg_train = y_train[:, 0] + y_train[:, 1] / 60.0
y_reg_test = y_test[:, 0] + y_test[:, 1] / 60.0

# define the custom mean absolute error function
def custom_mae(y_true, y_pred):
    differences = K.abs(y_true - y_pred)
    diff = K.mean(differences, axis=-1)
    return diff

# the common sense error for regression model
def common_sense_reg(y_true, y_pred):
    differences = K.abs(y_true - y_pred)
    diff = K.minimum(differences, tf.subtract(12.0, differences))
    return diff


# In[6]:


DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")
reg_model = keras.models.Sequential([
  DefaultConv2D(filters=64, kernel_size=7, input_shape=input_shape), # worse: kernel_size=3, filters=32
  keras.layers.MaxPooling2D(pool_size=2),
  DefaultConv2D(filters=128),
  DefaultConv2D(filters=128),
  keras.layers.MaxPooling2D(pool_size=2),
  DefaultConv2D(filters=256),
  DefaultConv2D(filters=256),
  keras.layers.MaxPooling2D(pool_size=2), 
  DefaultConv2D(filters=256), # add a 256-filter block here
  DefaultConv2D(filters=256),
  keras.layers.MaxPooling2D(pool_size=2),
  keras.layers.Flatten(),
  keras.layers.Dense(units=128, activation='relu'),
  keras.layers.Dropout(0.15), # better having this
  keras.layers.Dense(units=64, activation='relu'),
  keras.layers.Dropout(0.15),
  keras.layers.Dense(1, activation='linear'),
])


# In[7]:


print(reg_model.summary())


# In[8]:


# reg_model.compile(loss="mean_absolute_error", optimizer='sgd', metrics=[tf.keras.losses.MeanAbsoluteError()])
reg_model.compile(loss=custom_mae, optimizer='sgd', metrics=[common_sense_reg])

reg_history = reg_model.fit(X_train, y_reg_train, 
          batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_reg_test))
score = reg_model.evaluate(X_test, y_reg_test, verbose=0)
print('Test mean absolute error:', score[0])
print('Test common sense error:', score[1])


# In[9]:


reg_model.save("regression.h5")


# In[10]:


pd.DataFrame(reg_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 3.5)
plt.show()


# In[13]:


plt.plot(reg_history.history['loss'])
plt.plot(reg_history.history['val_loss'])
plt.plot(reg_history.history['common_sense_reg'])
plt.plot(reg_history.history['val_common_sense_reg'])
# plt.title('Regression')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train_mae', 'test_mae', 'train_cse', 'test_cse'], loc='upper right')
plt.show()


# ## Classification

# In[26]:


# basic parameter settings
batch_size = 64 # tuning this？
num_classes = 720 # 24 or 720
epochs = 60 # 24: 200， 720: 60

# change the two-integer labels to 24 or 720 classes
def classification_label(y, num_classes):
    label = []
    if num_classes == 24:
        for i in y:
            if i[1] <= 30:
                label.append(2 * i[0])
            else:
                label.append(2 * i[0] + 1)
    else: # num_classes == 720
        for i in y:
            label.append(60 * i[0] + i[1])
    return label

y_class_train = np.array(classification_label(y_train, num_classes))
y_class_test = np.array(classification_label(y_test, num_classes))

# convert class vectors to binary class matrices
y_class_train = keras.utils.to_categorical(y_class_train, num_classes)
y_class_test = keras.utils.to_categorical(y_class_test, num_classes)

# the common sense accuracy for classification model
def common_sense_class(y_true, y_pred):
    differences = K.abs(tf.math.argmax(y_true, 1) - tf.math.argmax(y_pred, 1))
    # num_classes = tf.cast(num_classes, tf.int64)
    differences = tf.cast(differences, tf.int32)
    diff = tf.math.minimum(differences, tf.subtract(num_classes, differences))
    return diff


# In[18]:


DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="same")
class_model_1 = keras.models.Sequential([
  DefaultConv2D(filters=32, input_shape=input_shape), # worse: kernel_size=3, filters=32
  keras.layers.MaxPooling2D(pool_size=2),
  keras.layers.BatchNormalization(),
  keras.layers.Dropout(0.2),
    
  DefaultConv2D(filters=64),
  DefaultConv2D(filters=64),
  keras.layers.MaxPooling2D(pool_size=2),
  keras.layers.BatchNormalization(),
  keras.layers.Dropout(0.2),
    
  # DefaultConv2D(filters=96),
  DefaultConv2D(filters=96),
  keras.layers.MaxPooling2D(pool_size=2),
  keras.layers.BatchNormalization(),
  keras.layers.Dropout(0.2),

  # DefaultConv2D(filters=128),
  DefaultConv2D(filters=128),
  keras.layers.MaxPooling2D(pool_size=2),
  keras.layers.BatchNormalization(),
  keras.layers.Dropout(0.2),

  keras.layers.Flatten(),
  keras.layers.Dense(units=256, activation='relu'),
  keras.layers.Dropout(0.2), # better having this
  keras.layers.Dense(units=128, activation='relu'),
  keras.layers.Dropout(0.2),
  keras.layers.Dense(units=num_classes, activation='softmax')
])


# In[19]:


class_model_1.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adadelta(learning_rate=0.1), metrics=["accuracy", common_sense_class])
# class_model_1.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adadelta(learning_rate=0.1), metrics=["accuracy"])

class_history_1 = class_model_1.fit(X_train, y_class_train,
        batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_class_test))
class_score_1 = class_model_1.evaluate(X_test, y_class_test, verbose=0)
print('Test loss:', class_score_1[0])
print('Test accuracy:', class_score_1[1])
print('Test common sense error:', class_score_1[2])


# In[20]:


class_model_1.save("Classification_24.h5")


# In[25]:


plt.plot(class_history_1.history['loss'])
plt.plot(class_history_1.history['val_loss'])
plt.plot(class_history_1.history['accuracy'])
plt.plot(class_history_1.history['val_accuracy'])
plt.plot(class_history_1.history['common_sense_class'])
plt.plot(class_history_1.history['val_common_sense_class'])
# plt.title('Classification_24')
# plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy', 'train_cse', 'test_cse'], loc='upper right')
plt.show()


# In[22]:


pd.DataFrame(class_history_1.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 3.5)
plt.show()


# In[27]:


class_model_2 = keras.models.Sequential([
  DefaultConv2D(filters=32, input_shape=input_shape), # worse: kernel_size=3, filters=32
  keras.layers.MaxPooling2D(pool_size=2),
  keras.layers.BatchNormalization(),
  keras.layers.Dropout(0.2),
    
  DefaultConv2D(filters=64),
  DefaultConv2D(filters=64),
  keras.layers.MaxPooling2D(pool_size=2),
  keras.layers.BatchNormalization(),
  keras.layers.Dropout(0.2),
    
  DefaultConv2D(filters=96), #
  DefaultConv2D(filters=96),
  keras.layers.MaxPooling2D(pool_size=2),
  keras.layers.BatchNormalization(),
  keras.layers.Dropout(0.2),

  DefaultConv2D(filters=128), #
  DefaultConv2D(filters=128),
  keras.layers.MaxPooling2D(pool_size=2),
  keras.layers.BatchNormalization(),
  keras.layers.Dropout(0.2),

  keras.layers.Flatten(),
  keras.layers.Dense(units=256, activation='relu'), #
  keras.layers.Dropout(0.2), # better having this
  keras.layers.Dense(units=128, activation='relu'), #
  keras.layers.Dropout(0.2),
  keras.layers.Dense(units=num_classes, activation='softmax')
])


# In[28]:


class_model_2.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adadelta(learning_rate=0.08), metrics=["accuracy", common_sense_class])
# class_model_2.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=[common_sense_class])

class_history_2 = class_model_2.fit(X_train, y_class_train,
        batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_class_test))
class_score_2 = class_model_2.evaluate(X_test, y_class_test, verbose=0)
print('Test loss:', class_score_2[0])
print('Test accuracy:', class_score_2[1])


# In[29]:


class_model_2.save("Classification_720.h5")


# In[35]:


plt.plot(class_history_2.history['loss'])
plt.plot(class_history_2.history['val_loss'])
plt.plot(class_history_2.history['accuracy'])
plt.plot(class_history_2.history['val_accuracy'])
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy'], loc='upper right')
plt.show()


# In[34]:


# plt.plot(class_history_2.history['loss'])
# plt.plot(class_history_2.history['val_loss'])
# plt.plot(class_history_2.history['accuracy'])
# plt.plot(class_history_2.history['val_accuracy'])
plt.plot(class_history_2.history['common_sense_class'])
plt.plot(class_history_2.history['val_common_sense_class'])
# plt.title('Classification_720')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train_cse', 'test_cse'], loc='upper right')
plt.show()


# ## Multi-head models

# In[4]:


# basic parameter settings
batch_size = 64
num_classes = 12
epochs = 200

# change the two-integer labels to two categories
y_hours_train, y_minutes_train = np.split(y_train, 2, 1)
y_hours_test, y_minutes_test = np.split(y_test, 2, 1)

# convert hour vectors to binary hour matrices
y_hours_train = keras.utils.to_categorical(y_hours_train, num_classes)
y_hours_test = keras.utils.to_categorical(y_hours_test, num_classes)


# In[6]:


DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=(3, 3), activation="relu", padding="same")

img_input = keras.layers.Input(shape=input_shape)
block1_conv1 = DefaultConv2D(filters=32)(img_input)
block1_pool = keras.layers.MaxPooling2D(pool_size=2)(block1_conv1)
block1_norm = keras.layers.BatchNormalization()(block1_pool)
block1_drop = keras.layers.Dropout(0.2)(block1_norm)

block2_conv1 = DefaultConv2D(filters=64)(block1_drop)
block2_conv2 = DefaultConv2D(filters=64)(block2_conv1)
block2_pool = keras.layers.MaxPooling2D(pool_size=2)(block2_conv2)
block2_norm = keras.layers.BatchNormalization()(block2_pool)
block2_drop = keras.layers.Dropout(0.2)(block2_norm)

block3_conv1 = DefaultConv2D(filters=96)(block2_drop)
block3_pool = keras.layers.MaxPooling2D(pool_size=2)(block3_conv1)
block3_norm = keras.layers.BatchNormalization()(block3_pool)
block3_drop = keras.layers.Dropout(0.2)(block3_norm)

block4_conv1 = DefaultConv2D(filters=128)(block3_drop)
block4_pool = keras.layers.MaxPooling2D(pool_size=2)(block4_conv1)
block4_norm = keras.layers.BatchNormalization()(block4_pool)
block4_drop = keras.layers.Dropout(0.2)(block4_norm)

flatten = keras.layers.Flatten()(block4_drop)
fc1 = keras.layers.Dense(units=256, activation="relu")(flatten)
drop1 = keras.layers.Dropout(0.2)(fc1)
fc2 = keras.layers.Dense(128, activation="relu")(drop1)
drop2 = keras.layers.Dropout(0.2)(fc2)

hour_class = keras.layers.Dense(num_classes, activation="softmax", name="hour_class")(drop2)
minute_reg = keras.layers.Dense(1, activation='linear', name="minute_reg")(drop2)
multi_model = keras.models.Model(inputs=[img_input], outputs=[hour_class, minute_reg])


# In[7]:


multi_model.compile(loss={'hour_class': "categorical_crossentropy", 'minute_reg': "mean_absolute_error"},
                    loss_weights={'hour_class': 1.0, 'minute_reg': 0.2},
                    optimizer=keras.optimizers.Adadelta(learning_rate=0.1), 
                    metrics={'hour_class': "accuracy", 'minute_reg': []})
# multi_model.compile(loss=["categorical_crossentropy", custom_mae], optimizer=keras.optimizers.Adadelta(learning_rate=0.1), 
#               metrics=["accuracy", common_sense_class, common_sense_reg]) # optimizer=[keras.optimizers.Adadelta(learning_rate=0.1), "sgd"]

multi_history = multi_model.fit(X_train, (y_hours_train, y_minutes_train), batch_size=batch_size, epochs=epochs, verbose=1, 
                    validation_data=(X_test, (y_hours_test, y_minutes_test)))
score = multi_model.evaluate(X_test, (y_hours_test, y_minutes_test), verbose=0)
print('Test loss:', score[0])
print('Test hour loss:', score[1])
print('Test minute loss:', score[2])
print('Test hour accuracy:', score[3])
# print('Test minute MeanAbsoluteError:', score[4])


# In[8]:


multi_model.save("Multi_head.h5")


# In[9]:


print(multi_history.history.keys())


# In[13]:


plt.plot(figsize=(8, 5))
plt.plot(multi_history.history['hour_class_loss'])
plt.plot(multi_history.history['val_hour_class_loss'])
plt.plot(multi_history.history['hour_class_accuracy'])
plt.plot(multi_history.history['val_hour_class_accuracy'])
plt.title('Output_hour')
# plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy'], loc='upper right')
plt.show()


# In[17]:


plt.plot(multi_history.history['minute_reg_loss'])
plt.plot(multi_history.history['val_minute_reg_loss'])
plt.plot(multi_history.history['loss'])
plt.plot(multi_history.history['val_loss'])
plt.title('Output_minute')
# plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_minute_loss', 'test_minute_loss', 'train_total_loss', 'test_total_loss'], loc='upper right')
plt.show()


# In[19]:


plt.plot(multi_history.history['minute_reg_loss'])
plt.plot(multi_history.history['val_minute_reg_loss'])
# plt.plot(multi_history.history['loss'])
# plt.plot(multi_history.history['val_loss'])
plt.title('Output_minute')
# plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'], loc='upper right')
plt.show()


# In[20]:


plt.plot(figsize=(8, 5))
plt.plot(multi_history.history['hour_class_loss'])
plt.plot(multi_history.history['val_hour_class_loss'])
plt.plot(multi_history.history['hour_class_accuracy'])
plt.plot(multi_history.history['val_hour_class_accuracy'])
plt.plot(multi_history.history['minute_reg_loss'])
plt.plot(multi_history.history['val_minute_reg_loss'])
plt.plot(multi_history.history['loss'])
plt.plot(multi_history.history['val_loss'])
plt.title('Output_hour')
# plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_hour_loss', 'test_hour_loss', 'train_hour_accuracy', 'test_hour_accuracy', 'train_minute_loss', 'test_minute_loss', 'train_total_loss', 'test_total_loss'], loc='upper right')
plt.show()


# In[12]:


pd.DataFrame(multi_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 10)
plt.show()


# In[18]:


# the common sense accuracy for Multi-head model
y_pred = multi_model.predict(X_test)
prediction = np.concatenate([np.argmax(y_pred[0], 1).reshape(3600, 1), y_pred[1]], axis=1)
true = np.concatenate([np.argmax(y_hours_test, 1).reshape(3600, 1), y_minutes_test], axis=1)
y_true = true[:, 0] + true[:, 1] / 60.0
y_prediction = prediction[:, 0] + prediction[:, 1] / 60.0

def common_sense_multi(y_true, y_pred):
    differences = np.abs(y_true - y_pred)
    diff = np.minimum(differences, np.subtract(12.0, differences))
    diff = np.mean(diff, axis=-1)
    return diff

cse = common_sense_multi(y_true, y_prediction)
print(cse)


# In[ ]:




