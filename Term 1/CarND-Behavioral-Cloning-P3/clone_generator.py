import csv, cv2, sklearn
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, ELU, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

channels, row, col = 3, 160, 320  # camera format
BATCH_SIZE=32
DIR='./driving_data/'
IMAGE_DIR=DIR+'IMG/'
CORRECTION = 0.3

# Import data
samples = []
with open(DIR+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:                            
                #center image               
                center = IMAGE_DIR+batch_sample[0].split('/')[-1]                
                center_image = cv2.imread(center)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                #left image                
                left = IMAGE_DIR+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left)
                left_angle = float(batch_sample[3])+CORRECTION
                images.append(left_image)
                angles.append(left_angle)
                #right image                
                right = IMAGE_DIR+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right)
                right_angle = float(batch_sample[3])-CORRECTION
                images.append(right_image)
                angles.append(right_angle)
                # if steering angle above threshold              
                if abs(float(center_angle)) > 0.1:
                    #center                    
                    ci_flipped = cv2.flip(center_image, 1)
                    ca_flipped = float(center_angle) * -1.0
                    images.append(ci_flipped)
                    angles.append(ca_flipped)
                    #left                    
                    li_flipped = cv2.flip(left_image, 1)
                    la_flipped = float(left_angle) * -1.0
                    images.append(li_flipped)
                    angles.append(la_flipped)
                    #left                    
                    ri_flipped = cv2.flip(right_image, 1)
                    ra_flipped = float(right_angle) * -1.0
                    images.append(ri_flipped)
                    angles.append(ra_flipped)
                    for i in range(5):
                        images.append(center_image)
                        angles.append(center_angle)
                        images.append(left_image)
                        angles.append(left_angle)
                        images.append(right_image)
                        angles.append(right_angle)
                        images.append(ci_flipped)
                        angles.append(ca_flipped)
                        images.append(li_flipped)
                        angles.append(la_flipped)
                        images.append(ri_flipped)
                        angles.append(ra_flipped)
                if abs(float(center_angle)) > 1.0:
                    for i in range(10):
                        images.append(center_image)
                        angles.append(center_angle)
                        images.append(left_image)
                        angles.append(left_angle)
                        images.append(right_image)
                        angles.append(right_angle)
                        images.append(ci_flipped)
                        angles.append(ca_flipped)
                        images.append(li_flipped)
                        angles.append(la_flipped)
                        images.append(ri_flipped)
                        angles.append(ra_flipped)
                if abs(float(center_angle)) > 1.5:
                    for i in range(20):
                        images.append(center_image)
                        angles.append(center_angle)
                        images.append(left_image)
                        angles.append(left_angle)
                        images.append(right_image)
                        angles.append(right_angle)
                        images.append(ci_flipped)
                        angles.append(ca_flipped)
                        images.append(li_flipped)
                        angles.append(la_flipped)
                        images.append(ri_flipped)
                        angles.append(ra_flipped)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)            

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)
            
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
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)


#model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')








