import csv, cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, ELU, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D

channels, row, col = 3, 160, 320  # camera format

# Import data
lines = []
with open("./driving_data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:    
    for i in range(3):
        source_path = line[i]
        tokens = source_path.split('/')
        filename = tokens[-1]
        localpath = "./driving_data/IMG/" + filename
        image = cv2.imread(localpath)
        images.append(image)
    correction = 0.25
    measurement = line[3]
    measurements.append(measurement)
    measurements.append(float(measurement)+correction)
    measurements.append(float(measurement)-correction)

assert len(images) == len(measurements)

augmented_images=[]
augmented_measurements=[]

for image, measurement in zip(images, measurements):
    if measurement > 1.0:
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        flipped_image = cv2.flip(image, 1)
        flipped_measurement = float(measurement) * -1.0
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)    

X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)
    
def get_leNet_model():
    
    model = Sequential()       
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, channels)))
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, channels)))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(Dropout(0.35))
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

model = get_leNet_model()
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')









