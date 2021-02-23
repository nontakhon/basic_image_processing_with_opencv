import cv2
import numpy as np

img = cv2.imread('./img/lottery_001.jpg')
cv2.imshow('Original', img)

# Converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

# Converting to grayscale
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv', hsv)

# Blur
blur = cv2.GaussianBlur(img, (7, 7), 0)
cv2.imshow('Blur', blur)

# Edge Cascade
canny = cv2.Canny(blur, 200, 200)
cv2.imshow('Canny Edges', canny)




cv2.waitKey(0)