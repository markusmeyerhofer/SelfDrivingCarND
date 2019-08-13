#!/usr/bin/env python
import cv2
import numpy as np

image = cv2.imread('./augmented_images/GREEN/TL_GREEN_0_0.jpeg')
print('Shape in: ', str(image.shape))
#im = cv2.resize(image, (100, 150)).astype(np.float32)
#m[:,:,0] -= 103.939
#im[:,:,1] -= 116.779
#im[:,:,2] -= 123.68
#im = image.transpose((2,0,1))
#im = np.expand_dims(im, axis=0)

#image = cv2.imwrite('./test_images/133936.jpg', im)


x = np.array(image)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
print('Shape out: ', str(x.shape))