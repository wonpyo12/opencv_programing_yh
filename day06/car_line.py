import cv2 as cv
import numpy as np

# 1단계: 이미지 로드
img = cv.imread('road.png')
#print(f"원본 이미지 크기 :{img.shape}") # 실제 크기 확인 

scale = 0.2 
img_resized = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
print(f"축소된 이미지 크기 :{img_resized.shape}") # 실제 크기 확인 
img_resized = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

# 1. BGR 이미지를 HLS 색상 공간으로 변환
hls = cv.cvtColor(img_resized, cv.COLOR_BGR2HLS)

# 2. 흰색 추출 범위 설정 (밝기 L이 아주 높은 영역)
lower_white = np.array([0, 200, 0], dtype=np.uint8)
upper_white = np.array([180, 255, 255], dtype=np.uint8)
mask_white = cv.inRange(hls, lower_white, upper_white)

# 3. 노란색 추출 범위 설정 (색상 H가 15~35 근처인 영역)
lower_yellow = np.array([15, 30, 115], dtype=np.uint8)
upper_yellow = np.array([35, 204, 255], dtype=np.uint8)
mask_yellow = cv.inRange(hls, lower_yellow, upper_yellow)

# 4. 흰색 마스크와 노란색 마스크를 합치기 (OR 연산)
color_mask = cv.bitwise_or(mask_white, mask_yellow)

# 5. 원본 이미지에 마스크 씌우기 (흰색, 노란색만 남고 나머지는 까맣게 변함)
masked_img = cv.bitwise_and(img_resized, img_resized, mask=color_mask)
gray = cv.cvtColor(masked_img, cv.COLOR_BGR2GRAY)

# 2단계: Canny 에지 검출 (day04.md 참고)
edges = cv.Canny(gray, 100, 200, apertureSize=3)

# 3단계: 허프 직선 변환
#lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=70, maxLineGap=10)

# 4단계: 검출된 직선을 원본 이미지에 그리기
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv.imshow('Original', gray)
cv.imshow('Edges', edges)
cv.imshow('Hough Lines', img_resized)
cv.waitKey(0)