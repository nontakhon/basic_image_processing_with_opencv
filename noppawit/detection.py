import cv2
import numpy as np

img = cv2.imread("./img/lottery_003.jpg")
resized = cv2.resize(img, (700, 500), interpolation=cv2.INTER_CUBIC)
cropped = resized[40:370, 330:600]
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
_, binary_img = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
kernel_pad = np.ones((5, 5), np.uint8)
print(kernel_pad)
dilated = cv2.dilate(binary_img, kernel_pad, iterations=3)

# show image
cv2.imshow('Resized', resized)
cv2.imshow('Cropped', cropped)
cv2.imshow('binary_img', binary_img)
cv2.imshow('Dilated', dilated)
cv2.waitKey(0)

# Find contours
contours, h = cv2.findContours(dilated, 1, 2)

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
