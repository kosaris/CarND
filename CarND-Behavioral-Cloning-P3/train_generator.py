import csv
import cv2
import numpy as np
import tensorflow as tf
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D, Lambda, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

samples = []
with open('data/my_driving4/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
        
correction = 0.2

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image_center = cv2.imread(batch_sample[0])
                image_left = cv2.imread(batch_sample[1])
                image_right = cv2.imread(batch_sample[2])
                 
                images.append(image_center)
                images.append(image_left)
                images.append(image_right)
                 
                angle = float(batch_sample[3])
                 
                angles.append(angle)
                angles.append(angle + correction)
                angles.append(angle - correction)
                 
                image_flipped_center = np.fliplr(image_center)
                image_flipped_left = np.fliplr(image_left)
                image_flipped_right = np.fliplr(image_right)
                 
                images.append(image_flipped_center)
                images.append(image_flipped_left)
                images.append(image_flipped_right)
                 
                angles.append(angle)
                angles.append(-(angle+correction))
                angles.append(-(angle-correction))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
batch_size = 64
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

# lines = []
# 
# with open('data/my_driving4/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         lines.append(line)
# 
# images = []
# measurements = []
# correction = 0.2
# for line in lines:
#     image_center = cv2.imread(line[0])
#     image_left = cv2.imread(line[1])
#     image_right = cv2.imread(line[2])
#     
#     images.append(image_center)
#     images.append(image_left)
#     images.append(image_right)
#     
#     measurement = float(line[3])
#     
#     measurements.append(measurement)
#     measurements.append(measurement + correction)
#     measurements.append(measurement - correction)
#     
#     image_flipped_center = np.fliplr(image_center)
#     image_flipped_left = np.fliplr(image_left)
#     image_flipped_right = np.fliplr(image_right)
#     
#     images.append(image_flipped_center)
#     images.append(image_flipped_left)
#     images.append(image_flipped_right)
#     
#     measurements.append(measurement)
#     measurements.append(-(measurement+correction))
#     measurements.append(-(measurement-correction))
# 
# X_train = np.array(images)
# y_train = np.array(measurements)

#history_object = model.fit(X_train, y_train, validation_split = .2, shuffle=True, nb_epoch=5)


# build the convolution network with 4 conv2d layers and 4 fully connected layers
model = Sequential()
model.add(Lambda(lambda x:x/255.0 - .5 , input_shape= (160, 320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(3,160,320)))
model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(.5))
model.add(Dense(50))
model.add(Dropout(.5))
model.add(Dense(10))
model.add(Dropout(.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)/batch_size, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=7)
#model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)

model.save('model9.h5')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

