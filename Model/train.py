# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 15:34:35 2020

@author: krishna
"""


import keras.backend as k
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation, Reshape
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import CSVLogger
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from PIL import Image
import pickle 
import random
import tensorflow as tf
from model1 import resnet
#to save images to background, we will use 'AGG' setting of matplotlib
matplotlib.use('Agg')

#path to dataset
dataset = 'data'

#path to save model
model_path = 'model.h5'

#path to save labels
label_path = '/'

#path to save plots
plot_path='/'

HP_LR = 1e-3
HP_EPOCHS = 5
HP_IMAGE_DIM = (225,225,3)
HP_BS = 64
data = []
classes = []
ptrain = '/content/drive/My Drive/Project/Data/train'

for root, dirs, files in os.walk(ptrain):
  print((root.split(os.path.sep)[-1])[2:])
  for each in files:
    try:
      path=root+'/'+each
      im = Image.open(path)  
      width, height = im.size
      if(width>=225 and height>=225):   
        left = (width - 225)/2
        top = (height - 225)/2
        right = (width + 225)/2
        bottom = (height + 225)/2
        im1 = im.crop((left, top, right, bottom))
        newsize = (225, 225)
        im1 = im1.resize(newsize) 
        image_array = img_to_array(im1)
        x= np.array(image_array,dtype='float')/255.0    
        data.append(x)
        label = (root.split(os.path.sep)[-1])[2:]
        classes.append(label)
    except Exception as e:
      print(e)
print('done with images')
labels = np.array(classes)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
data=np.asarray(data)
print(data.shape,labels.shape)
#print(labels)

xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.2,random_state=42)
 
aug = ImageDataGenerator(rotation_range=0.25,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

classifier =resnet.build(height=225,width=225,depth=3,classes=len(lb.classes_))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum = 0.9, nesterov = True)
opt = Adam(lr=HP_LR, decay=HP_LR/HP_EPOCHS)
classifier.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
csv_logger = CSVLogger('training.log')

checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')

hist = classifier.fit_generator(aug.flow(xtrain,ytrain,batch_size=HP_BS), validation_data=(xtest,ytest),steps_per_epoch=len(xtrain)//HP_BS,epochs=HP_EPOCHS,callbacks=[csv_logger,checkpoint])


classifier.save('mymodel1.h5')