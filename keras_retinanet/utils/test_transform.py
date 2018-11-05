__package__ = 'keras_retinanet.utils'
from .transform import random_transform
from .image import apply_transform, TransformParameters, adjust_transform_for_image
import cv2

image = cv2.imread('test.jpg')
cv2.imshow('image', image)
cv2.waitKey(0)
m1 = random_transform(flip_x_chance=1)
print(m1)
transform_parameters = TransformParameters()
m2 = adjust_transform_for_image(m1, image, False)
print(m2)
transformed = apply_transform(m2, image, transform_parameters)
cv2.imshow('transformed', transformed)
cv2.waitKey(0)
