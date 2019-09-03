import numpy as np
from cv2 import imread
from os import walk
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('../resources/crocodiles_imgs/img1.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
new_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        new_contours.append(cnt)
print("Number of new contours=" + str(len(new_contours)))
print(new_contours[0])

cv2.drawContours(img, new_contours, -1, (0, 255, 0), 10)
cv2.imshow("Image", img)
cv2.imshow("Img_gray", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
