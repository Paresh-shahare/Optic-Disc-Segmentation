import tensorflow
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import *
from keras.layers.merge import *
import keras
from keras.optimizers import Adam

%matplotlib inline

width = 512
height = 512

Training_input_data = r'C:/Users/Paresh/Downloads/dataset/train/origin1/'
training_input_data = [x for x in sorted(os.listdir(Training_input_data))]
print(len(training_input_data))
print(training_input_data)
x_train_input_data = np.empty((len(training_input_data),width,height),dtype = 'float32')
for i,name  in enumerate(training_input_data):
    im = cv2.imread(Training_input_data + name,cv2.COLOR_BGR2GRAY).astype('int16').astype('float32')/255.
    im = im[:,:,2]
    im = cv2.resize(im,dsize = (width,height),interpolation = cv2.INTER_NEAREST)
    x_train_input_data[i] = im
    
print(x_train_input_data.shape)
fig, ax = plt.subplots(1,2,figsize = (8,4))
ax[0].imshow(x_train_input_data[0])
ax[1].imshow(x_train_input_data[1],cmap='gray')
x_train_input_data = x_train_input_data.reshape(x_train_input_data.shape[0],width,height,1)
print(x_train_input_data.shape)


Testing_input_data = r'C:/Users/Paresh/Downloads/dataset/validate/origin1/'
testing_input_data = [x for x in sorted(os.listdir(Testing_input_data))]
print(len(testing_input_data))
print(testing_input_data)
x_test_input_data = np.empty((len(testing_input_data),width,height),dtype = 'float32')
for i,name  in enumerate(testing_input_data):
    im = cv2.imread(Testing_input_data + name,cv2.COLOR_BGR2GRAY).astype('int16').astype('float32')/255.
    img = im[:,:,2]
    img = cv2.resize(img,dsize = (width,height),interpolation = cv2.INTER_NEAREST)
    x_test_input_data[i] = img
    
print(x_test_input_data.shape)
fig, ax = plt.subplots(1,2,figsize = (8,4))
ax[0].imshow(x_test_input_data[0])
ax[1].imshow(x_test_input_data[1],cmap = 'gray')
x_test_input_data = x_test_input_data.reshape(x_test_input_data.shape[0],width,height,1)
print(x_test_input_data.shape)
print(x_test_input_data[0])




Training_output_data = r'C:/Users/Paresh/Downloads/dataset/train/groundtruth1/'
training_output_data = [x for x in sorted(os.listdir(Training_output_data))]
print(len(training_output_data))
print(training_output_data)
x_train_output_data = np.empty((len(training_output_data),width,height),dtype = 'float32')
for i,name  in enumerate(training_output_data):
    im = cv2.imread(Training_output_data + name).astype('int16').astype('float32')/255.
    img = im[:,:,2]
    img = cv2.resize(img,dsize = (width,height),interpolation = cv2.INTER_NEAREST)
    x_train_output_data[i] = img
    
print(x_train_output_data.shape)
fig, ax = plt.subplots(1,2,figsize = (8,4))
ax[0].imshow(x_train_output_data[0])
ax[1].imshow(x_train_output_data[1],cmap='gray')
x_train_output_data = x_train_output_data.reshape(x_train_output_data.shape[0],width,height,1)
print(x_train_output_data.shape)


Testing_output_data = r'C:/Users/Paresh/Downloads/dataset/validate/groundtruth1/'
testing_output_data = [x for x in sorted(os.listdir(Testing_output_data))]
print(len(testing_output_data))
print(testing_output_data)
x_test_output_data = np.empty((len(testing_output_data),width,height),dtype = 'float32')
for i,name  in enumerate(testing_output_data):
    im = cv2.imread(Testing_output_data + name).astype('int16').astype('float32')/255.
    img = im[:,:,2]
    img = cv2.resize(img,dsize = (width,height),interpolation = cv2.INTER_NEAREST)
    x_test_output_data[i] = img
    
print(x_test_output_data.shape)
fig, ax = plt.subplots(1,2,figsize = (8,4))
ax[0].imshow(x_test_output_data[0])
ax[1].imshow(x_test_output_data[1],cmap='gray')
x_test_output_data = x_test_output_data.reshape(x_test_output_data.shape[0],width,height,1)
print(x_test_output_data.shape)


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c




def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((width, height, 1))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model


model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()

model.fit(x_train_input_data,x_train_output_data,
          batch_size = 1,
          epochs = 15,
          verbose = 1,
          validation_data = (x_test_input_data,x_test_output_data))

model.save("UNetW_final.h5")