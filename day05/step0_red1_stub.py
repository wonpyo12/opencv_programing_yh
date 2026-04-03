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

areat = 1000
# --- stub 함수: 아직 구현하지 않음 ---
def nothing(x):
    pass
cv.namedWindow('Trackbar')
cv.createTrackbar('H_min', 'Trackbar', 35, 179, nothing)
cv.createTrackbar('H_max', 'Trackbar', 85, 179, nothing)
cv.createTrackbar('S_min', 'Trackbar', 50, 255, nothing)
cv.createTrackbar('S_max', 'Trackbar', 255, 255, nothing)
cv.createTrackbar('V_min', 'Trackbar', 50, 255, nothing)
cv.createTrackbar('V_max', 'Trackbar', 255, 255, nothing)


def detect_color(frame):
    h_min = cv.getTrackbarPos('H_min', 'Trackbar')
    h_max = cv.getTrackbarPos('H_max', 'Trackbar')
    s_min = cv.getTrackbarPos('S_min', 'Trackbar')
    s_max = cv.getTrackbarPos('S_max', 'Trackbar')
    v_min = cv.getTrackbarPos('V_min', 'Trackbar')
    v_max = cv.getTrackbarPos('V_max', 'Trackbar')
    
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    """특정 색상 감지 (GREEN에서 구현할 예정)"""
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv,lower,upper)
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