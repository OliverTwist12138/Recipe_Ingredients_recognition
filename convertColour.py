import cv2
import numpy as np
path = 'fruits-360_dataset/fruits-360/Training/Apricot/0_100.jpg'
img = cv2.imread(path)
# print(image)
# mean = [0.6840, 0.5786, 0.5037]
# std = [0.3035, 0.3600, 0.3914]
# image = np.array(image)
# img = image * np.array(std) + np.array(mean)
# cv2.imshow('img',img)
# cv2.waitKey(0)
hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, S, V = cv2.split(hsvImg)
newImg = cv2.merge([H,S,V,greyImg])
cv2.imshow('img',newImg)
cv2.waitKey(0)