import cv2
import cv2 as cv
import urllib.request

from sampledownload import get_sample
import numpy as np

# 방법 1: 강사 제공 샘플 이미지 사용

img = cv2.imread(get_sample('run.png'))
scale = 0.1 
img_resized = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
print(f"축소된 이미지 크기 :{img_resized.shape}") # 실제 크기 확인 
img_resized = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))


# 방법 2: 인터넷에서 다운로드

# URL에서 이미지를 받아와 저장

# url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg"

# 또는 다른 보행자 사진 URL 사용 가능

# urllib.request.urlretrieve(url, 'pedestrian.jpg')

# img = cv2.imread('pedestrian.jpg')

if img is None:

    print("Error: 이미지 로드 실패")

else:

    print(f"✅ 이미지 로드 성공: {img.shape}")

    cv2.imshow('Original', img_resized)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
hog = cv2.HOGDescriptor()
# ② 사전학습된 보행자 검출 모델 로드 ---

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ③ 이미지에서 보행자 검출 ---

detections, weights = hog.detectMultiScale(

    img_resized,

    winStride=(8, 8),      # 스캔 윈도우 이동 크기

    padding=(16, 16),      # 윈도우 주변 패딩

    scale=1.05             # 이미지 피라미드 스케일

)

print(f"검출된 보행자: {len(detections)}명")

# ④ 신뢰도 분석 ---

if len(weights) > 0:

    print(f"신뢰도 범위: {weights.min():.3f} ~ {weights.max():.3f}")

    print(f"신뢰도별 개수:")

    for threshold in [0.3, 0.5, 0.7, 0.9]:

        count = np.sum(weights > threshold)

        print(f"  weights > {threshold}: {count}명")

# ⑤ 검출 결과 시각화 ---

result = img_resized.copy()

for (x, y, w, h) in detections:

    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Pedestrian Detection', result)

cv2.waitKey(0)

cv2.destroyAllWindows()
# 신뢰도 필터링

CONFIDENCE_THRESHOLD = 0.5  # 임계값 조정 (0.3 ~ 0.9)

result_filtered = img_resized.copy()

filtered_count = 0

for (x, y, w, h), weight in zip(detections, weights):

    if weight > CONFIDENCE_THRESHOLD:

        cv2.rectangle(result_filtered, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 신뢰도 표시

        cv2.putText(result_filtered, f'{weight:.2f}', (x, y-5),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        filtered_count += 1

print(f"\n신뢰도 필터링 (threshold={CONFIDENCE_THRESHOLD}):")

print(f"  필터링 전: {len(detections)}명")

print(f"  필터링 후: {filtered_count}명")

print(f"  감소율: {(len(detections)-filtered_count)/len(detections)*100:.1f}%")

cv2.imshow('Filtered Detection', result_filtered)

cv2.waitKey(0)

cv2.destroyAllWindows()
configs = [

    {"name": "Default", "winStride": (8, 8), "scale": 1.05},

    {"name": "Fine-grained", "winStride": (4, 4), "scale": 1.05},

    {"name": "Small objects", "winStride": (8, 8), "scale": 1.02},

    {"name": "Fast", "winStride": (16, 16), "scale": 1.1},

]

print("\n파라미터별 검출 결과:")

print("-" * 60)

for config in configs:

    found, _ = hog.detectMultiScale(

        img_resized,

        winStride=config["winStride"],

        padding=(16, 16),

        scale=config["scale"]

    )

    print(f"{config['name']:15s} (winStride={config['winStride']}, scale={config['scale']}): {len(found):3d}명")

