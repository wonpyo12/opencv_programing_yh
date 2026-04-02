import urllib.request
import os
import numpy as np 
import cv2 as cv 

def get_sample(filename, repo='insightbook'):

    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

# 사용 방법
# img = cv.imread(get_sample('morphological.png', repo='insightbook'))

img = cv.imread(get_sample('messi5.jpg', repo='opencv'))

h,w = img.shape[:2]
 
M = cv.getRotationMatrix2D(((w-1)/2.0,(h-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(w,h))
 
cv.imshow('Rotation',dst)
cv.waitKey(0)
cv.destroyAllWindows()