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
canny = cv2.Canny(blur, 125, 175)
cv2.imshow('Canny Edges', canny)

# thresh
_, binary_img = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('thresh', binary_img)

kernel = np.ones((5, 5), np.uint8)
print(kernel)

# Dilating the image
dilated = cv2.dilate(binary_img, kernel, iterations=3)
cv2.imshow('Dilated', dilated)

kernel = np.ones((2, 1), np.uint8)
print(kernel)

# Eroding
eroded = cv2.erode(canny, kernel, iterations=3)
cv2.imshow('Eroded', eroded)

# Resize
resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
cv2.imshow('Resized', resized)

# Cropping
cropped = img[50:200, 200:400]
cv2.imshow('Cropped', cropped)

cv2.imwrite("blur_image.jpg", blur)

cv2.waitKey(0)
