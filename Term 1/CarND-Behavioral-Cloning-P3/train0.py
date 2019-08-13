import csv, cv2, sklearn
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, ELU, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

channels, row, col = 3, 160, 320  # camera format
BATCH_SIZE=32

# Import data
samples = []
with open('./driving_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './driving_data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
        

def get_nvidia_model():
    # NVIDIA End to End Learning Pipeline Model

    model = Sequential()

    # Conv layer 1, 5x5 kernel to 24@ (from 3@)
    # TODO: What does the number of filters MEAN?
    # Stride of 2x2
    model.add(Convolution2D(24, 5, 5, input_shape=(row, col, channels)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    # Conv layer 2, 5x5 kernel to 36@
    model.add(Convolution2D(36, 5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    # Conv layer 3, 5x5 kernel to 48@
    model.add(Convolution2D(48, 5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    # Conv layer 4, 3x3 kernel to 64@
    model.add(Convolution2D(64, 3, 3))  
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    # Conv layer 5, 3x3 kernel to 64@
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    # Flatten
    model.add(Flatten())

    # Fully connected layer 1, 1164 neurons
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    # Fc2, 100 neurons
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    # Fc3, 50 neurons
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    # Fc4, 10 neurons
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    # Output
    model.add(Dense(1))

    # Compile model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
              
    return model

def get_coreai_model(time_len=1):

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, channels), output_shape=(row, col, channels)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model
    
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
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)


#model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')



#model.add(Cropping2D(cropping=((70,25),(0,0))))








