#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import backend as K

batch_size = 32
epochs = 15

img_width, img_height = 32, 32
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'train_images',  # this is the target directory
        target_size=(32, 32),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'test_images',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=5000 // batch_size,
        epochs=3,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)


model.save('./keras_model_new.h5')

import cv2, os
def load_image(image_path):

    image = cv2.imread(image_path) 
    image = cv2.resize(image,(32,32), interpolation = cv2.INTER_CUBIC)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #image = normalize_image(image)

    '''
    # scale image (0-255)
    if image_path[-4:] == '.png':
        image = image.astype(np.float32)*255
    '''

    # remove alpha channel if present
    if image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        image = np.dstack((r,g,b))

    return image

def get_classification(image):

        choices = {0: "GREEN", 1: "YELLOW", 2: "RED", 3: "UNKNOWN"}
        res = None
        res = cv2.resize(image, (32,32), interpolation = cv2.INTER_CUBIC)
        image = res.reshape(1, 32, 32, 3)
        classification = model.predict_classes(image, verbose=0)[0]
        result = choices.get(classification, 'UNKNOWN')
        return  result

eval_green = True
eval_yellow = True
eval_red = True
verbose_mode = False


dir = './test_images'
green_file_path = dir + '/GREEN/'
yellow_file_path = dir + '/YELLOW/'
red_file_path = dir + '/RED/'

num_images = 0
num_incorrect = 0

if eval_green:
    green_images=os.listdir(green_file_path) 
    for green_image_path in green_images:
        if type(green_image_path)==type("string"):
            image = load_image(green_file_path + green_image_path)
            result = get_classification(image)
            num_images += 1
            if result != 'GREEN':
                num_incorrect += 1
            if verbose_mode:
                print('Expected GREEN - detected: ', result)

if eval_yellow:
    yellow_images=os.listdir(yellow_file_path) 
    for yellow_image_path in yellow_images:
        if type(yellow_image_path)==type("string"):
            image = load_image(yellow_file_path + yellow_image_path)
            result = get_classification(image)
            num_images += 1
            if result != 'YELLOW':
                num_incorrect += 1
            if verbose_mode:
                print('Expected YELLOW - detected: ', result)

if eval_red:
    red_images=os.listdir(red_file_path) 
    for red_image_path in red_images:
        if type(red_image_path)==type("string"):
            image = load_image(red_file_path + red_image_path)
            result = get_classification(image)
            num_images += 1
            if result != 'RED':
                num_incorrect += 1
            if verbose_mode:
                print('Expected RED - detected: ', result)

if (eval_green or eval_yellow or eval_red):
    print(' ')
    print('==================================================================================================')
    print(' ')
    print('No Images: ' + str(num_images) + ' incorrect: ' + str(num_incorrect) + ' success rate: ' + str(100.0-100.0*(float(num_incorrect)/float(num_images))) + ' %')
    print(' ')
    print('==================================================================================================')
    print(' ')