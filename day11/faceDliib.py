import cv2
import dlib
import numpy as np

# 1. Dlib 얼굴 검출기 및 68 특징점 추출기 로드
# 주의: 'shape_predictor_68_face_landmarks.dat' 파일이 같은 폴더에 있어야 합니다.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
print("68 특징점 추출 시작... (q로 종료)")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 얼굴 영역 검출 (Cascade 대신 dlib 사용)
    faces = detector(gray)
    
    for face in faces:
        # 3. 검출된 얼굴 영역에서 68개의 특징점 추출
        landmarks = predictor(gray, face)
        
        # 4. 특징점 68개를 화면에 초록색 점으로 그리기 (실습 확인용)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow('Face Landmarks', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()