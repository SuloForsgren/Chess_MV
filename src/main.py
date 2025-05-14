import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('img/origin.jpg', cv.IMREAD_GRAYSCALE)
blur = cv.GaussianBlur(img, (5,5), 0)
edges = cv.Canny(blur, 50, 150)

contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
img_color = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

for point in contours:
    approx = cv.approxPolyDP(point, 0.02 * cv.arcLength(point, True), True)
    area = cv.contourArea(point)

    if len(approx) == 4 :
        x, y, w, h = cv.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.9 < aspect_ratio < 1.1 :
            largest_square = approx

if largest_square is not None:
    cv.drawContours(img_color, [largest_square], -1, (0, 255, 0), 2)


plt.figure(figsize=(12, 10))
plt.imshow(img_color)
plt.title('Contours on Edge Image')
plt.xticks([]), plt.yticks([])
plt.show()