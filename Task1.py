#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from functools import partial


# ## Importing Data
# 
# First, we import the data and split it in a train and test set. We further split the training data of the fashion_mnist dataset into a validation and training set.

# In[2]:


(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
plt.imshow(X_train_full[0])


# In[3]:


X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


# In[4]:


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(class_names[y_train_full[0]])
X_train_full.shape


# ## MLP implementation
# 
# In this section, we use a sequential model on the fashion_mnist dataset.

# ### Original model from book
# 
# The following code comes from the textbook, we will use this model as a starting point. Later, we will try different hyperparameters for this model.

# In[54]:


model = Sequential([
 keras.layers.Flatten(input_shape=[28, 28]),
 keras.layers.Dense(300, activation="relu"),
 keras.layers.Dense(100, activation="relu"),
 keras.layers.Dense(10, activation="softmax")
])


# In[55]:


model.summary()


# In[56]:


model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])


# In[57]:


history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


# In[58]:


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
print("Accuracy", history.history["accuracy"][-1:])


# As can be seen, the performance is not that great yet. So, we will change some hyperparameters to see how this influences the performance.
# 
# ### Hyperparameter testing/tuning

# In[59]:


# Extra layers
model2 = Sequential([
 keras.layers.Flatten(input_shape=[28, 28]),
 keras.layers.Dense(300, activation="relu"),
 keras.layers.Dense(100, activation="relu"),
 keras.layers.Dense(100, activation="relu"),
 keras.layers.Dense(10, activation="softmax")
])

# softplus
model3 = Sequential([
 keras.layers.Flatten(input_shape=[28, 28]),
 keras.layers.Dense(300, activation="softplus"),
 keras.layers.Dense(100, activation="softplus"),
 keras.layers.Dense(10, activation="softmax")
])

# tanh and L1
model4 = Sequential([
 keras.layers.Flatten(input_shape=[28, 28]),
 keras.layers.Dense(300, activation="tanh", kernel_regularizer='l1'),
 keras.layers.Dense(100, activation="relu"),
 keras.layers.Dense(10, activation="softmax")
])

# sigmoid and dropout
model5 = Sequential([
 keras.layers.Flatten(input_shape=[28, 28]),
 keras.layers.Dense(300, activation="sigmoid"),
 keras.layers.Dropout(0.2),
 keras.layers.Dense(100, activation="sigmoid"),
 keras.layers.Dense(10, activation="softmax")
])

# L1 and L2
model6 = Sequential([
 keras.layers.Flatten(input_shape=[28, 28]),
 keras.layers.Dense(300, activation="relu", kernel_regularizer='l2'),
 keras.layers.Dense(100, activation="relu", kernel_regularizer='l1'),
 keras.layers.Dense(10, activation="softmax")
])

# L1L2
model7 = Sequential([
 keras.layers.Flatten(input_shape=[28, 28]),
 keras.layers.Dense(300, activation="relu", kernel_regularizer='l1_l2'),
 keras.layers.Dense(100, activation="relu"),
 keras.layers.Dense(10, activation="softmax")
])


# In[60]:


model2.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model3.compile(loss="sparse_categorical_crossentropy", optimizer="Adamax", metrics=["accuracy"])
model4.compile(loss="sparse_categorical_crossentropy", optimizer= keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])
model5.compile(loss="sparse_categorical_crossentropy", optimizer= keras.optimizers.Adam(learning_rate=0.5), metrics=["accuracy"])
model6.compile(loss="sparse_categorical_crossentropy", optimizer= keras.optimizers.SGD(learning_rate=0.01), metrics=["accuracy"])
model7.compile(loss="sparse_categorical_crossentropy", optimizer= keras.optimizers.SGD(learning_rate=0.1), metrics=["accuracy"])


# In[72]:


history2 = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


# In[73]:


history3 = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


# In[63]:


history4 = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


# In[64]:


history5 = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


# In[65]:


history6 = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


# In[66]:


history7 = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


# In[74]:


print('Validation accuracies at final epoch of each model: ')
for x in [history, history2, history3, history4, history5, history6, history7]:
    pd.DataFrame(x.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    plt.show()


# ## CNN implementation
# 
# Next, we use a CNN on the same dataset and try out different parameters

# ### Original model from book
# 
# The below code shows the original model as described in the textbook. Just like before, this model is used as the base model. Using this model, we will try different hyperparameters. But now we replace it with selu instead of relu activation function.

# In[5]:


DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")

cnn = keras.models.Sequential([
 DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
 keras.layers.MaxPooling2D(pool_size=2),
 DefaultConv2D(filters=128),
 DefaultConv2D(filters=128),
 keras.layers.MaxPooling2D(pool_size=2),
 DefaultConv2D(filters=256),
 DefaultConv2D(filters=256),
 keras.layers.MaxPooling2D(pool_size=2),
 keras.layers.Flatten(),
 keras.layers.Dense(units=128, activation='selu'),
 keras.layers.Dropout(0.5),
 keras.layers.Dense(units=64, activation='selu'),
 keras.layers.Dropout(0.5),
 keras.layers.Dense(units=10, activation='softmax'),
])


# In[6]:


cnn.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])


# In[7]:


cnnhis = cnn.fit(X_train, y_train, epochs=18, validation_data=(X_valid, y_valid))


# In[ ]:


pd.DataFrame(cnnhis.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
print("Accuracy", cnnhis.history["accuracy"][-1:])
print("Val-Accuracy", cnnhis.history["val_accuracy"][-1:])
print("Loss", cnnhis.history["loss"][-1:])
print("Val-Loss", cnnhis.history["val_loss"][-1:])


# Model is overfitting, time to add regularisation.

# In[8]:


DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")

cnn = keras.models.Sequential([
 DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
 keras.layers.MaxPooling2D(pool_size=2),
 DefaultConv2D(filters=128),
 DefaultConv2D(filters=128),
 keras.layers.MaxPooling2D(pool_size=2),
 DefaultConv2D(filters=256),
 DefaultConv2D(filters=256),
 keras.layers.MaxPooling2D(pool_size=2),
 keras.layers.Flatten(),
 keras.layers.Dense(units=128, activation='selu'),
 keras.layers.Dense(units=64, activation='selu', kernel_regularizer='l2'),
 keras.layers.Dense(units=10, activation='softmax'),
])


# In[9]:


cnn.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
cnn.summary()


# In[10]:


cnnhis = cnn.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


# In[11]:


pd.DataFrame(cnnhis.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
print("Accuracy", cnnhis.history["accuracy"][-1:])
print("Val-Accuracy", cnnhis.history["val_accuracy"][-1:])
print("Loss", cnnhis.history["loss"][-1:])
print("Val-Loss", cnnhis.history["val_loss"][-1:])


# In[18]:


cnn.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
cnn.summary()
cnnhis = cnn.fit(X_train, y_train, epochs=3, validation_data=(X_valid, y_valid))


# In[19]:


pd.DataFrame(cnnhis.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
print("Accuracy", cnnhis.history["accuracy"][-1:])
print("Val-Accuracy", cnnhis.history["val_accuracy"][-1:])
print("Loss", cnnhis.history["loss"][-1:])
print("Val-Loss", cnnhis.history["val_loss"][-1:])


# # CiFAR10
# 

# In[55]:


(X_traincif, y_traincif), (X_testcif, y_testcif) = cifar10.load_data()


# In[56]:


X_traincif = X_traincif/255
X_testcif = X_testcif/255
X_testcif[1][16]


# In[57]:


print(X_traincif.shape)
X_validCif, X_trainCif = X_traincif[:5000], X_traincif[5000:]
y_validCif, y_trainCif = y_traincif[:5000], y_traincif[5000:]
print(X_validCif.shape)
print(X_trainCif.shape)


# In[58]:


outputs = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# In[ ]:


out = outputs
for t in range(10):
    plt.imshow(X_testcif[t])
    plt.title(outputs[int(y_testcif[t])])
    plt.show()


# ## MLP for CiFAR 10
# 

# In[25]:


model = Sequential([
 keras.layers.Flatten(input_shape=[32, 32, 3]),
 keras.layers.Dense(3072, activation="sigmoid"),
 keras.layers.Dense(1024, activation="sigmoid"),
 keras.layers.Dense(768, activation="sigmoid"),
 keras.layers.Dense(10, activation="softmax")
])
model.summary()


# In[26]:


optimizer = keras.optimizers.Adamax
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[27]:


history = model.fit(X_trainCif, y_trainCif, epochs=10, validation_data=(X_validCif, y_validCif))


# In[28]:


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
print("Accuracy", history.history["accuracy"][-1:])
print("Val-Accuracy", history.history["val_accuracy"][-1:])
print("Loss", history.history["loss"][-1:])
print("Val-Loss", history.history["val_loss"][-1:])


# ## CNN

# In[74]:


DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")

cnnModel = keras.models.Sequential([
 DefaultConv2D(filters=64, kernel_size=7, input_shape=[32, 32, 3]),
 keras.layers.MaxPooling2D(pool_size=2),
 DefaultConv2D(filters=128),
 DefaultConv2D(filters=128),
 keras.layers.MaxPooling2D(pool_size=2),
 DefaultConv2D(filters=256),
 DefaultConv2D(filters=256),
 keras.layers.MaxPooling2D(pool_size=2),
 keras.layers.Flatten(),
 keras.layers.Dense(units=1024, activation='selu'),
 keras.layers.Dense(units=256, activation='selu'),
 keras.layers.Dense(units=64, activation='selu'),
 keras.layers.Dense(units=10, activation='softmax'),
])

cnnModel.summary()


# In[75]:


cnnModel.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[76]:


cnnHistory = cnnModel.fit(X_trainCif, y_trainCif, epochs=18, validation_data=(X_validCif, y_validCif))


# In[78]:


pd.DataFrame(cnnHistory.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
print("Accuracy", cnnHistory.history["accuracy"][-1:])
print("Val-Accuracy", cnnHistory.history["val_accuracy"][-1:])
print("Loss", cnnHistory.history["loss"][-1:])
print("Val-Loss", cnnHistory.history["val_loss"][-1:])


# In[79]:


cnnModel.evaluate(X_testcif, y_testcif)


# Well that is a lot better, but what about black and white transformed.

# ## Custom Architecture
# I found a few architectures on google scholar to see if it works. The first one is from (https://www.researchgate.net/publication/326816043_FAWCA_A_Flexible-greedy_Approach_to_find_Well-tuned_CNN_Architecture_for_Image_Recognition_Problem)

# In[112]:


cnnModel = keras.Sequential(layers=[
    keras.layers.Conv2D(filters=96, kernel_size=5, input_shape=[32, 32, 3], padding="SAME"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(filters=80, kernel_size=5, padding="SAME", kernel_regularizer='l2'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(filters=96, kernel_size=5, padding="SAME", kernel_regularizer='l2'),
    keras.layers.Conv2D(filters=64, kernel_size=5, padding = "SAME", kernel_regularizer='l2'),
    keras.layers.Flatten(),
    keras.layers.Dense(units=256, activation="relu", kernel_regularizer='l2'),
    keras.layers.Dense(units=64, activation="relu", kernel_regularizer='l2'),
    keras.layers.Dense(units=10, activation="softmax")
])

cnnModel.summary()


# In[113]:


cnnModel.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
cnnHistory = cnnModel.fit(X_trainCif, y_trainCif, epochs=64, validation_data=(X_validCif, y_validCif))


# In[114]:


pd.DataFrame(cnnHistory.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 2) # set the vertical range to [0-1]
plt.show()
print("Accuracy", cnnHistory.history["accuracy"][-1:])
print("Val-Accuracy", cnnHistory.history["val_accuracy"][-1:])
print("Loss", cnnHistory.history["loss"][-1:])
print("Val-Loss", cnnHistory.history["val_loss"][-1:])
cnnModel.evaluate(X_testcif, y_testcif)

