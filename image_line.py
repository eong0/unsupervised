import cv2
from matplotlib import pyplot as plt

img = cv2.imread('test_img.jpeg')

cv2.line(img, (10,1400), (2300,1400), (0,0,0))
# 원점 (150,150), 반지름100
#cv2.circle(img, (1500,600), 100, (255,0,0))

#cv2. imshow('circle',img)
#cv2.waitKey(0)
plt.imshow(img)
