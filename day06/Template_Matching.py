import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os

def get_sample(filename, repo='insightbook'):
    """부교 또는 OpenCV 공식 샘플 이미지 자동 다운로드
    
    Args:
        filename (str): 이미지 파일명 (예: 'morphological.png')
        repo (str): 'insightbook' (부교) 또는 'opencv' (공식)
    
    Returns:
        str: 다운로드된 파일명
    """
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename



img = cv.imread(get_sample('sudoku.png'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)

# 선분 방식: (x1, y1, x2, y2) 시작점과 끝점으로 표현
lines = cv.HoughLinesP(edges, 1, np.pi/180, 
                       threshold=20, minLineLength=50, maxLineGap=10)

# 검출된 직선 그리기
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv.imshow('Hough Lines P', img)
cv.waitKey(0)
cv.destroyAllWindows()