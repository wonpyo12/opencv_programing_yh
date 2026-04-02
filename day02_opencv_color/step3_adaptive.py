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

cv.createTrackbar('blockSize',window_name,11,31,nothing)
cv.createTrackbar('C',window_name,2,20,nothing)

while True:
    block_size = cv.getTrackbarPos('blockSize',window_name)
    c_val = cv.getTrackbarPos('C',window_name)
    
    if block_size < 3 :
        block_size =3
    if block_size % 2== 0:
        block_size+=1
    _,th_global = cv.threshold(img,127,255,cv.THRESH_BINARY)
    _,th_otsu = cv.threshold(img,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

    th_mean =cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                  cv.THRESH_BINARY, block_size, c_val)
    th_gaussian = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY, block_size, c_val)
    top = np.hstack([th_global, th_otsu])
    bottom = np.hstack([th_mean, th_gaussian])
    result = np.vstack([top, bottom])
    result_size = cv.resize(result, None, fx=0.7, fy=0.5, interpolation=cv.INTER_AREA)
    cv.imshow(window_name,result_size)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()