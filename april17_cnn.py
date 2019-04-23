# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:57:00 2019

@author: a

Building the xyz training and predicting
"""

# Importing the Keras libraries and packages
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from livelossplot import PlotLossesKeras
# Initialising the CNN
class An_xyz():
    global xyz
    batch_size=40
    xyz = Sequential()
# Step 1 - Convolution
    xyz.add(Conv2D(32, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
# Step 2 - Pooling
    xyz.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
    xyz.add(Conv2D(48 , (3, 3), activation = 'relu'))
    xyz.add(MaxPooling2D(pool_size = (2, 2)))
#Adding third convolutional layer
    xyz.add(Conv2D(32, (3, 3), activation = 'relu'))
    xyz.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
    xyz.add(Flatten())
# Step 4 - Full connection
    xyz.add(Dense(units = 2, activation = 'relu'))
    xyz.add(Dense(units = 2, activation = 'softmax'))
# Compiling the CNN
    xyz.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory('C:\\Users\\anany\\OneDrive\\Desktop\\an_python\\dataset\\face_dataset_train')
    test_set = test_datagen.flow_from_directory('C:\\Users\\anany\\OneDrive\\Desktop\\an_python\\dataset\\face_dataset_test')
    xyz.fit_generator(training_set,steps_per_epoch =400//batch_size,epochs = 10,callbacks=[PlotLossesKeras()],validation_data =test_set,validation_steps = 200//batch_size)
# Part 3 - Making new predictions
    x=training_set.class_indices
        
model_json = xyz.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
xyz.save_weights("model.h5")
print("Saved model to disk")
