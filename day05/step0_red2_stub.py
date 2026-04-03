import cv2 as cv
import numpy as np
import serial
try:
    ser = serial.Serial('COM6', 9600, timeout=1) 
    print("✅ 아두이노와 연결되었습니다.")
except Exception as e:
    print(f"❌ 시리얼 연결 실패: {e}")
    ser = None
LOWER_COLOR = np.array([100, 70, 30])
UPPER_COLOR = np.array([140, 255, 255])
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([35, 255, 255])


areat = 1000
# --- stub 함수: 아직 구현하지 않음 ---


def detect_color(frame):
    
    """특정 색상 감지 (GREEN에서 구현할 예정)"""
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    mask_blue = cv.inRange(hsv,LOWER_COLOR,UPPER_COLOR)
    mask_yellow = cv.inRange(hsv,LOWER_YELLOW,UPPER_YELLOW)
    mask = cv.bitwise_or(mask_blue,mask_yellow)
    area  = cv.countNonZero(mask)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel,iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel,iterations=2)
    result = area > areat
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    max_contour =None
    max_area = 0
    bbox =None
    for cnt in contours:
        area = cv.contourArea(cnt) # 윤곽선의 면적 계산
        if area > areat and area > max_area:
            max_area = area
            max_contour = cnt

    # 가장 큰 윤곽선을 찾았다면 사각형 정보를 가져옵니다.
    if max_contour is not None:
        # boundingRect: 윤곽선을 감싸는 최소 크기의 곧은 사각형 반환
        bbox = cv.boundingRect(max_contour)

    return bbox,mask,max_area


cap = cv.VideoCapture(2)

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다")
    exit()
box_history = []
MAX_HISTORY = 5
detected = False
while True :
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다")
        cap.release()
        exit()
    
# detect_color() 함수가 아직 구현되지 않았으므로 False 반환
    result ,mask,area= detect_color(frame)

# RED 단계: 이 부분이 실행되어야 함 (즉, FAIL이 출력되어야 함)
    if result is not None:
        box_history.append(result)
        if len(box_history) > MAX_HISTORY:
            box_history.pop(0)
        avg_x = int(sum(b[0] for b in box_history) / len(box_history))
        avg_y = int(sum(b[1] for b in box_history) / len(box_history))
        avg_w = int(sum(b[2] for b in box_history) / len(box_history))
        avg_h = int(sum(b[3] for b in box_history) / len(box_history))
        cv.rectangle(frame, (avg_x, avg_y), (avg_x + avg_w, avg_y + avg_h), (0, 255, 0), 3)
        cv.putText(frame, f"DETECTED (Area: {int(area)})", (avg_x, avg_y - 10), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if not detected:
            if ser is not None:
                ser.write(b'1')
            detected = True
    else:
       box_history = []
       cv.putText(frame, "SEARCHING...", (30, 50), 
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
       if detected:
            if ser is not None:
                ser.write(b'0')
            detected = False    
    cv.imshow("cap",frame)
    cv.imshow("Color Mask", mask)
    if cv.waitKey(1) & 0xFF == ord('q'):  # 'q' 누르면 종료
       
        break

cap.release()
cv.destroyAllWindows()