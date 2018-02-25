import csv
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D


lines = []

with open('data/my_driving3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
correction = 0.2
for line in lines:
    image_center = cv2.imread(line[0])
    image_left = cv2.imread(line[1])
    image_right = cv2.imread(line[2])
    
    images.append(image_center)
    images.append(image_left)
    images.append(image_right)
    
    measurement = float(line[3])
    
    measurements.append(measurement)
    measurements.append(measurement + correction)
    measurements.append(measurement - correction)
    
    image_flipped_center = np.fliplr(image_center)
    image_flipped_left = np.fliplr(image_left)
    image_flipped_right = np.fliplr(image_right)
    
    images.append(image_flipped_center)
    images.append(image_flipped_left)
    images.append(image_flipped_right)
    
    measurements.append(measurement)
    measurements.append(-(measurement+correction))
    measurements.append(-(measurement-correction))

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - .5 , input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(3,160,320)))
model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = .2, shuffle=True, nb_epoch=7)

model.save('model6.h5')