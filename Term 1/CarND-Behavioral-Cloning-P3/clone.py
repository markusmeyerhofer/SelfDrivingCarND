import csv, cv2
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D

channels, row, col = 3, 160, 320  # camera format
DIR = './data/'
IMG_DIR=DIR+"IMG/"
# Import data
lines = []
with open(DIR+"driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:    
    measurement = line[3]    
    if abs(float(measurement)) > 0.0001:
        for i in range(3):
            source_path = line[i]
            tokens = source_path.split('/')
            filename = tokens[-1]
            localpath = IMG_DIR + filename
            image = cv2.imread(localpath)
            images.append(image)
        correction = 0.2
        measurements.append(measurement)
        measurements.append(float(measurement)+correction)
        measurements.append(float(measurement)-correction)

assert len(images) == len(measurements)

augmented_images=[]
augmented_measurements=[]

print("Number of initial images: ", len(images))

for image, measurement in zip(images, measurements):
    #if abs(float(measurement)) > 0.2:
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = float(measurement) * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    inflation_factor=math.ceil(abs(   ((float(measurement)*10.0)**2)  ))
    for i in range(inflation_factor):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)

print("Number of images: ", len(augmented_images))

X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)

def get_model():
    
    model = Sequential()       
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, channels)))
    model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(row, col, channels)))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))    
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')   
    return model

model = get_model()
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')
