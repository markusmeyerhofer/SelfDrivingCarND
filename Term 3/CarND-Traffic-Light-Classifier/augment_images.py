#!/usr/bin/env python
from keras.preprocessing.image import ImageDataGenerator
import cv2, os
import numpy as np

def load_img():
    image_path = './sim_images/GREEN/35500.jpg'
    image = cv2.imread(image_path)
    return image

# MAIN
datagen = ImageDataGenerator(
        rotation_range=3,
        width_shift_range=0.05,
        height_shift_range=0.05,
        rescale=1./255,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest')

indir = './sim_images'
green_file_path = indir + '/GREEN/'
yellow_file_path = indir + '/YELLOW/'
red_file_path = indir + '/RED/'

green_images=os.listdir(green_file_path) 
for green_image_path in green_images:
    if type(green_image_path)==type("string"):
        image = cv2.imread(green_file_path + green_image_path)
        x = np.array(image)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='./augmented_images/GREEN', save_prefix='TL_GREEN', save_format='jpeg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely

yellow_images=os.listdir(yellow_file_path) 
for yellow_image_path in yellow_images:
    if type(yellow_image_path)==type("string"):
        image = cv2.imread(green_file_path + green_image_path)
        x = np.array(image)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='./augmented_images/YELLOW/', save_prefix='TL_YELLOW', save_format='jpeg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely

red_images=os.listdir(red_file_path) 
for red_image_path in red_images:
    if type(red_image_path)==type("string"):
        image = cv2.imread(green_file_path + green_image_path)
        x = np.array(image)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='./augmented_images/RED/', save_prefix='TL_RED', save_format='jpeg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely