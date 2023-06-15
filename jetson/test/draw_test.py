
import jetson.utils
import numpy as np
import cv2

source = jetson.utils.videoSource("file://data/sample.jpeg")

rgb_img = source.Capture()
img_width = rgb_img.width
img_height = rgb_img.height

left = int(0.5*img_width)
top = int(0.5*img_height)
right = int(0.6*img_width)
bottom = int(0.6*img_height)

color = (0,255,127,200)

jetson.utils.cudaDrawRect(rgb_img, (left, top, right, bottom), color)

file_out = "data/sample_rect.jpeg"
# save the image
jetson.utils.cudaDeviceSynchronize()
jetson.utils.saveImage(file_out, rgb_img)
print("saved {:d}x{:d} test image to '{:s}'".format(img_width, img_height, file_out))
