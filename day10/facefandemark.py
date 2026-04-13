import cv2
import dlib
import numpy as np
from imutils import face_utils 

# 오렌지 이미지 로드 
orange_img = cv2.imread('orange.jpg')
orange_img = cv2.resize(orange_img, (512, 512))

# dlib : 얼굴 감지기 초기화 
detector = dlib.get_frontal_face_detector()

# dlib : 랜드마크 예측기 초기화 (모델 파일 )
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat')

# 웹캠 시작 
#cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture(0) 
def get_roi(img,points) :
    x.y,w,h = cv2.boundingRect(points)
    roi = img[y:y+h, x:x+w]
    return roi,(x,y,w,h)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (512, 512))

    # 그레이 스케일 변환 (얼굴 감지는 색상 불필요)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = detector(gray)

    if len(faces) == 0:
        #얼굴이 없으면 원본 프레임 출력 
        cv2.imshow('result', frame)
        continue

    for face in faces:
        # 얼굴 영역에서 68개의 랜드마크 좌표 예측 
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape) # Numpy 배열로 변환  

        # 눈 영역 
        left_eye = shape[36:42] # 왼쪽 눈 
        right_eye = shape[42:48] # 오른쪽 눈 
        mouth = shape[48:58] # 입  
    
    # 랜드마크 점 전체를 화면에 그려보기
    for (x, y) in shape:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    cv2.imshow('landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()