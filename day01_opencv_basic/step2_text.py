import numpy as np
import cv2 as cv
 
# Create a black image
img = cv.imread("./my_photo.png")
h,w = img.shape[:2] 
overLay =img.copy()


cv.rectangle(overLay, (0, h - 200), (w, h), (0, 0, 0), -1)
cv.addWeighted(overLay, 0.5, img, 0.5, 0, img)


font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'YH',(10,350), font, 1,(255,255,255),2,cv.LINE_AA)
cv.putText(img,'wonpyo',(10,400), font, 1,(255,255,255),2,cv.LINE_AA)

cv.imshow("Drawing image",img)
cv.waitKey(0)
cv.destroyAllWindows