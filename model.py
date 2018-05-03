import os
import csv

samples = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            # Augmentation: flip left and right
            augmented_images,augmented_measurements =[],[]
            for image, measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*(-1))
        
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Train and validation generators
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x : x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0)))) # 65x320x3
model.add(Convolution2D(3, 5, 5, activation="relu")) # 61x316x3
model.add(MaxPooling2D((2,2))) # 30x158x3
model.add(Convolution2D(6, 5, 5, activation="relu")) # 26x154x6
model.add(MaxPooling2D((2,2))) # 13x77x6

model.add(Convolution2D(18, 5, 5, activation="relu")) #9x73x18
model.add(MaxPooling2D((2,2))) # 16x36x18
model.add(Convolution2D(48, 3, 3, activation="relu")) #4x34x48
model.add(MaxPooling2D((2,2))) # 2x17x48
model.add(Convolution2D(132, 1, 1, activation="relu")) #2x17x132
#model.add(Dropout(0.5))

model.add(Flatten())
model.add (Dense(120))
model.add (Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')

model.fit_generator(train_generator, samples_per_epoch= len(2*train_samples),\
                    validation_data=validation_generator, \
                    nb_val_samples=len(validation_samples), nb_epoch=10, verbose=1)
#model.fit(X_train,y_train,validation_split =0.2, shuffle =True, nb_epoch=7, verbose=1)
model.save('model.h5')
