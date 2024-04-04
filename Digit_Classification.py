#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

get_ipython().system('pip install tensorflow')
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')


# In[3]:


df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')
df_test.head()


# In[4]:


df_train.head()


# In[5]:


# DATA PRE-PROCESSING:


# In[54]:


def preprocessing(df_train, df_test):
    X_train = df_train.drop('label', axis=1)
    X_train = np.array(X_train)
    X_train = X_train / 255.0
    
    X_test = np.array(df_test)
    X_test = X_test / 255.0
    
    y_train = np.array(df_train['label'])
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, shuffle=True, random_state=42)

    return X_train, X_val, X_test, y_train, y_val

X_train, X_val, X_test, y_train, y_val = preprocessing(df_train, df_test)


# In[55]:


y_train


# In[56]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y_train = pd.get_dummies(y_train).values
y_val = pd.get_dummies(y_val).values


# In[57]:


y_val


# In[58]:


# DATA AUGMENTATION:


# In[59]:


train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)


# In[50]:


train_datagen


# In[10]:


# MODELING AND TRAINING:


# In[60]:


batch_size = 32
classes = 10
n_epochs = 30
n_steps = round(len(X_train)/100)
verbosity = 1


# In[45]:


# M1:


# In[61]:


model_1 = Sequential()
model_1.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model_1.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'))
model_1.add(MaxPool2D(pool_size=(2, 2), strides=2, padding="valid"))
model_1.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', activation = 'relu'))
model_1.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', activation = 'relu'))
model_1.add(MaxPool2D(pool_size=(2, 2), strides=2, padding="valid"))
model_1.add(Flatten())
model_1.add(Dense(120, activation='relu'))
model_1.add(Dense(10, activation='softmax'))


# In[62]:


model_1.build()
model_1.summary()
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[63]:


history_1 = model_1.fit(X_train,y_train, epochs=n_epochs, verbose=verbosity, validation_data=(X_val, y_val), steps_per_epoch=n_steps)


# In[65]:


y_pred = model_1.predict(X_val)


# In[66]:


Y_pred = np.argmax(y_pred,axis=1)


# In[67]:


Y_val = np.argmax(y_val,axis=1)


# In[68]:


from sklearn.metrics import accuracy_score


# In[69]:


accuracy_score(Y_pred,Y_val)


# In[ ]:


# M2:


# In[70]:


model_2 = Sequential()
model_2.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model_2.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_2.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation = 'relu'))
model_2.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_2.add(Flatten())
model_2.add(Dense(120, activation='relu'))
model_2.add(Dense(84, activation='relu'))
model_2.add(Dense(10, activation='softmax'))


# In[71]:


model_2.build()
model_2.summary()
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[72]:


history_2 = model_2.fit(x=train_generator, epochs=n_epochs, verbose=verbosity, validation_data=(X_val, y_val), steps_per_epoch=n_steps)


# In[77]:


y_pred_2 = model_2.predict(X_val)
Y_pred_2 = np.argmax(y_pred_2,axis=1)
accuracy_score(Y_pred_2,Y_val)


# In[ ]:


#M3:


# In[74]:


model_3 = Sequential()
model_3.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model_3.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_3.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation = 'relu'))
model_3.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_3.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation = 'relu'))
model_3.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_3.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation = 'relu'))
model_3.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_3.add(Flatten())
model_3.add(Dense(120, activation='relu'))
model_3.add(Dense(84, activation='relu'))
model_3.add(Dense(10, activation='softmax'))


# In[75]:


model_3.build()
model_3.summary()
model_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[76]:


history_3 = model_3.fit(x=train_generator, epochs=n_epochs, verbose=verbosity, validation_data=(X_val, y_val), steps_per_epoch=n_steps)


# In[80]:


y_pred_3 = model_3.predict(X_val)
Y_pred_3 = np.argmax(y_pred_3,axis=1)
accuracy_score(Y_pred_3,Y_val)


# In[81]:


import seaborn as sns
sns.set_theme()


# In[82]:


plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history_1.history["accuracy"], label="model_1")
plt.plot(history_2.history["accuracy"], label="model_2")
plt.plot(history_3.history["accuracy"], label="model_3")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training accuracy")
plt.show()


# In[22]:


plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history_1.history["val_accuracy"], label="model_1")
plt.plot(history_2.history["val_accuracy"], label="model_2")
plt.plot(history_3.history["val_accuracy"], label="model_3")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Validation accuracy")
plt.show()


# In[ ]:


# OPTIMIZATION:


# In[ ]:


# EARLY STOPPING:


# In[83]:


callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)


# In[ ]:


# LEARNING RATE SCHEDULE:


# In[84]:


lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)


# In[ ]:


# OPTIMIZER:


# In[85]:


adam = keras.optimizers.Adam(learning_rate=lr_schedule)


# In[86]:


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', activation = 'relu'))
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', activation = 'relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[87]:


model.build()
model.summary()
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[88]:


history = model.fit(x=train_generator, epochs=n_epochs, verbose=verbosity, callbacks=[callback], validation_data=(X_val, y_val), steps_per_epoch=n_steps)


# In[89]:


y_pred_es = model.predict(X_val)
Y_pred_es = np.argmax(y_pred_es,axis=1)
accuracy_score(Y_pred_es,Y_val)


# In[90]:


plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs. Validation accuracy")
plt.show()


# In[ ]:


# PREDICTIONS:


# In[91]:


def plot_digit(X, i):
    plt.imshow(X[i].reshape(28, 28), cmap=plt.get_cmap('gray'))


# In[92]:


plot_digit(X_train, 5)


# In[93]:


y_pred = model.predict(X_test)
y_test = np.argmax(y_pred, axis=1)


# In[94]:


def make_predictions(model, i):
    print(f'Predicted Digit: {y_test[i]}')
    plot_digit(X_test, i)


# In[95]:


make_predictions(model_1, 4)

