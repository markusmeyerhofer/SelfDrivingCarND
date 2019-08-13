import csv, cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D

channels, row, col = 3, 160, 320  # camera format
DIR = './driving_data/'
IMG_DIR=DIR+"IMG/"
ELITIST_THRESHOLD=1.1 #(1.1)
INFLATION_FACTOR=0    #(100)
CORRECTION = 0.3

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
    #if abs(float(measurement)) > 0.001:
    for i in range(3):
        source_path = line[i]
        tokens = source_path.split('/')
        filename = tokens[-1]
        localpath = IMG_DIR + filename
        image = cv2.imread(localpath)
        images.append(image)
        if float(measurement) > ELITIST_THRESHOLD:
            for i in range(INFLATION_FACTOR):
                images.append(image)
    
    measurements.append(measurement)
    if float(measurement) > ELITIST_THRESHOLD:
            for i in range(INFLATION_FACTOR):
                measurements.append(measurement)
    measurements.append(float(measurement)+CORRECTION)
    if float(measurement) > ELITIST_THRESHOLD:
            for i in range(INFLATION_FACTOR):
                measurements.append(float(measurement)+CORRECTION)
    measurements.append(float(measurement)-CORRECTION)
    if float(measurement) > ELITIST_THRESHOLD:
            for i in range(INFLATION_FACTOR):
                measurements.append(float(measurement)-CORRECTION)  
    
assert len(images) == len(measurements)

print("Number of images: ", len(images))

X_train = np.array(images)
Y_train = np.array(measurements)

def get_model():
    
    model = Sequential()       
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, channels)))
    model.add(Cropping2D(cropping=((60,15), (0,0)), input_shape=(row, col, channels)))
    model.add(Convolution2D(24, 5, 5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    model.add(Convolution2D(36, 5, 5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    model.add(Convolution2D(48, 5, 5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')   
    return model

model = get_model()
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')
