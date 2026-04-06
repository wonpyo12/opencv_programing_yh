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

# 이미지 로드
import cv2 as cv
import numpy as np

# 이미지 로드
img = cv.imread(get_sample('messi5.jpg'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
template = gray[80:230, 20:150]

# Template Matching
result = cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)

# 임계값 이상의 모든 매칭 위치 찾기
threshold = 0.8  # 80% 이상 유사도
locations = np.where(result >= threshold)

# 모든 후보 표시
result_img = img.copy()
h, w = template.shape[:2]

for y, x in zip(locations[0], locations[1]):
    top_left = (x, y)
    bottom_right = (x + w, y + h)
    cv.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 1)

cv.imshow('All Matches Above Threshold', result_img)
cv.waitKey(0)
cv.destroyAllWindows()

print(f"Found {len(locations[0])} matches above {threshold} threshold")
