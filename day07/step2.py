import cv2 as cv

import numpy as np

import matplotlib.pyplot as plt

from sampledownload import get_sample

# ========== Step 1: 이미지 로드 ==========

# 공식 샘플 이미지 또는 자신의 이미지 사용

img1 = cv.imread('book.png', cv.IMREAD_GRAYSCALE)

img2 = cv.imread('bookimg.png', cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:

    print("Error: 이미지를 찾을 수 없습니다.")

    exit()

print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}")

# ========== Step 2: 특징점 검출기 초기화 ==========

# SIFT 또는 ORB 선택

# TODO: 아래 중 하나를 선택해서 코드 작성하세요

sift = cv.SIFT_create()# 1) SIFT (느리지만 정확):

# 2) ORB (빠르지만 덜 정확): sift = cv.ORB_create()

# ========== Step 3: 키포인트와 디스크립터 추출 ==========

kp1, des1 = sift.detectAndCompute(img1, None)

kp2, des2 = sift.detectAndCompute(img2, None)

# 위 두 줄을 구현하세요

print(f"Keypoints found - img1: {len(kp1)}, img2: {len(kp2)}")

# ========== Step 4: FLANN 매칭기 설정 ==========

# SIFT는 float descriptor → KDTREE

# ORB는 binary descriptor → LSH

# TODO: detector_type을 판단하고 FLANN 파라미터 설정하세요

# SIFT 사용 시:

FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

search_params = dict(checks=50)

# ORB 사용 시:

#   FLANN_INDEX_LSH = 6

#   index_params = dict(algorithm=FLANN_INDEX_LSH, 

#                       table_number=12, key_size=20, multi_probe_level=2)

#   search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

# ========== Step 5: knnMatch로 k=2 매칭 ==========

matches = flann.knnMatch(des1, des2, k=2)

print(f"Total matches: {len(matches)}")

# ========== Step 6: Lowe's 비율 테스트 ==========

good_matches = []

for match_pair in matches:

    if len(match_pair) == 2:

        m, n = match_pair


        # TODO: m.distance < 0.7 * n.distance 를 확인하고
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
        # 조건을 만족하면 good_matches에 추가하세요

print(f"Good matches after Lowe's ratio test: {len(good_matches)}")

# ========== Step 7: 시각화 ==========

if len(good_matches) >= 10:

    img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches,outImg=None, flags=2)
    plt.imshow(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))
    plt.title(f"Good Matches ({len(good_matches)})")
    cv.waitKey(0)
    plt.show()
else:

    print("Not enough good matches!")
MIN_MATCH_COUNT = 10
if len(good_matches) >= MIN_MATCH_COUNT:

    # TODO: 다음을 구현하세요

    # 1) good_matches에서 queryIdx와 trainIdx로 키포인트 좌표 추출
    
    # 2) np.float32 배열로 변환 (reshape(-1, 1, 2) 형태)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    

    # TODO: 호모그래피 행렬 계산 (RANSAC 사용)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    

    if M is not None:

        # TODO: img1의 네 모서리를 이미지 좌표 배열로 정의

        h, w = img1.shape

        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        

        # TODO: perspectiveTransform으로 변환된 좌표 계산

        dst = cv.perspectiveTransform(pts, M)

        

        # ========== Step 2: 결과 시각화 ==========

        # TODO: 다음을 구현하세요

        # 1) img2를 카피
        
        # 2) cv.polylines()로 변환된 사각형 그리기 (파란색, 두께 3)

        # 3) matplotlib으로 표시

        

        result_img = img2.copy()
        result_img = cv.polylines(result_img, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
        # TODO: polylines 코드 작성

        


        # ========== Step 3: 매칭 시각화 (inlier만) ==========

        matchesMask = mask.ravel().tolist()

        draw_params = dict(

            matchColor=(0, 255, 0),

            singlePointColor=None,

            matchesMask=matchesMask,

            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

        )
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        # 1) 배경 이미지에 파란색 네모 박스 그리기
        result_img = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        result_img = cv.polylines(result_img, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
        plt.subplot(1, 2, 1) # 1행 2열 중 첫 번째 칸
        plt.imshow(cv.cvtColor(img1, cv.COLOR_GRAY2RGB)) # 흑백 원본 출력
        plt.title('Original Image (book.png)')
        plt.axis('off')

        # 오른쪽 화면: 네모 박스 쳐진 배경 이미지
        plt.subplot(1, 2, 2) # 1행 2열 중 두 번째 칸
        plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB)) # 파란 상자 출력
        plt.title('Detected Object (No Lines!)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        

        inlier_count = sum(matchesMask)

        outlier_count = len(matchesMask) - inlier_count

        print(f"Inliers: {inlier_count}, Outliers: {outlier_count}")

    else:

        print("Failed to compute homography")

else:

    print(f"Not enough matches ({len(good_matches)}/{MIN_MATCH_COUNT})")
