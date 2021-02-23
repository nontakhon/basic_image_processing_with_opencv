import cv2
import numpy as np

img = cv2.imread("./img/lottery_001.jpg")
resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
cropped = resized[20:370, 230:420]
img_hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
print(img_hsv)

# HSV BLUE Color
color_min = np.array([60, 50, 50], np.uint8)
color_max = np.array([120, 255, 255], np.uint8)
threshold_img = cv2.inRange(img_hsv, color_min, color_max)
kernel_pad = np.ones((5, 5), np.uint8)
threshold_img = cv2.dilate(threshold_img, kernel_pad, iterations=4)


# HSV Black Color
color_min_b = np.array([0, 0, 50], np.uint8)
color_max_b = np.array([0, 255, 255], np.uint8)
threshold_img = cv2.inRange(img_hsv, color_min, color_max)
kernel_pad = np.ones((5, 5), np.uint8)
threshold_img_1 = cv2.dilate(threshold_img, kernel_pad, iterations=4)


cv2.imshow('hsv',img_hsv)
cv2.imshow('Resized', resized)
cv2.imshow('Cropped',cropped)
cv2.imshow('Dilated', threshold_img)
cv2.imshow('Dilated', threshold_img_1)



# Find contours
contours, h = cv2.findContours(threshold_img, 1, 2)

for cnt in contours:

    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    print(len(approx))

    if len(approx) == 5:
        print("pentagon")
        cv2.drawContours(cropped, [cnt], 0, 255, 3)

    elif len(approx) == 3:
        print("triangle")
        cv2.drawContours(cropped, [cnt], 0, (0, 255, 0), 3)

    elif len(approx) == 4:
        print("square")
        cv2.drawContours(cropped, [cnt], 0, (0, 0, 255), 3)

    elif len(approx) == 9:
        print("half-circle")
        cv2.drawContours(cropped, [cnt], 0, (255, 255, 0), 3)

    elif len(approx) < 15:
        print("polygon")
        cv2.drawContours(cropped, [cnt], 0, (255, 0, 255), 3)

    elif len(approx) > 15:
        print("circle")
        cv2.drawContours(cropped, [cnt], 0, (0, 255, 255), 3)

cv2.imwrite('result.jpg', resized)
cv2.imshow("result", resized)

cv2.waitKey(0)