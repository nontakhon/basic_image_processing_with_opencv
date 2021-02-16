import cv2
import numpy as np

# open new Mondrian Piet painting photo
img = cv2.imread("./img/lottery_001.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# threshold for hue channel in blue range
color_min = np.array([0, 200, 50], np.uint8)
color_max = np.array([10, 250, 255], np.uint8)
threshold_img = cv2.inRange(img_hsv, color_min, color_max)
cv2.imshow("color", threshold_img)
cv2.waitKey()
kernel_pad = np.ones((5, 5), np.uint8)
threshold_img = cv2.dilate(threshold_img, kernel_pad, iterations=3)

# show image
cv2.imshow('Dilated', threshold_img)
cv2.waitKey(0)

# Find contours
contours, h = cv2.findContours(threshold_img, 1, 2)

for cnt in contours:

    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    print(len(approx))

    if len(approx) == 5:
        print("pentagon")
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 1)

    elif len(approx) == 3:
        print("triangle")
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)

    elif len(approx) == 4:
        print("square")
        cv2.drawContours(img, [cnt], 0, (0, 0, 255), 3)

    elif len(approx) == 9:
        print("half-circle")
        cv2.drawContours(img, [cnt], 0, (255, 255, 0), 3)

    elif len(approx) < 15:
        print("polygon")
        cv2.drawContours(img, [cnt], 0, (255, 0, 255), 3)

    elif len(approx) > 15:
        print("circle")
        cv2.drawContours(img, [cnt], 0, (0, 255, 255), 3)

cv2.imwrite('result.jpg', img)
cv2.imshow("result", img)
cv2.waitKey(0)
