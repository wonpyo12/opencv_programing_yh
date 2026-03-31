import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import urllib.request
import os


def get_samples(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url,filename)
    return cv.imread(filename,0)
img = get_samples('sudoku.png')

assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.medianBlur(img,5)
def nothing(x):
    pass
window_name = 'sudoku'
cv.namedWindow(window_name)

cv.createTrackbar('threshold',window_name,127,255,nothing)
cv.createTrackbar('mode',window_name,0,1,nothing)

while True:
    threshold_val=cv.getTrackbarPos('threshold',window_name)
    mode_val = cv.getTrackbarPos('mode',window_name)
    if mode_val == 0 :
        current_mode=cv.THRESH_BINARY
    else :
        current_mode = cv.THRESH_BINARY_INV
    ret, result = cv.threshold(img,threshold_val,255,current_mode)

    cv.putText(result,f'Thresh:{threshold_val}',(10,40),cv.FONT_HERSHEY_SIMPLEX,1,255,2)
    combined = np.hstack([img, result])
    cv.imshow(window_name,combined)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()