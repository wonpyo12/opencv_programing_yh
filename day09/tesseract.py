import cv2
import numpy as np
import pytesseract

# 1. Tesseract 엔진 경로 설정 (본인 PC 환경에 맞게 유지)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 2. 이미지 불러오기 및 전처리
img = cv2.imread('01ga1134.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast = clahe.apply(gray)
binary = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cropped_plate = None

# 3. 번호판 영역만 가위로 오려내기 (Crop)
for contour in contours:
    area = cv2.contourArea(contour)
    if 2000 < area < 50000:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 2 < aspect_ratio < 6:
            # 번호판 영역만 잘라냄
            cropped_plate = binary[y:y+h, x:x+w]
            break

# 4. 잘라낸 이미지로 OCR 글자 인식 수행
if cropped_plate is not None:
    # 올려주신 사진처럼 검은 바탕에 흰 글씨(또는 반대)로 깔끔하게 반전
    clean_plate = cv2.bitwise_not(cropped_plate)
    
    # [핵심] Tesseract 옵션: psm 7 (이미지를 '한 줄의 텍스트'로 취급하여 번호판 인식률 극대화)
    custom_config = r'--oem 3 --psm 7'
    
    # 전체 텍스트 추출
    text = pytesseract.image_to_string(clean_plate, lang='kor+eng', config=custom_config)
    print(f"\n=============================")
    print(f" 최종 인식된 번호: {text.strip()}")
    print(f"=============================\n")

    # 글자별 상세 데이터 및 신뢰도 추출
    data = pytesseract.image_to_data(clean_plate, lang='kor+eng', config=custom_config, output_type=pytesseract.Output.DICT)
    
    print("[글자별 상세 인식 결과 및 신뢰도]")
    confidences = [] # 사진처럼 신뢰도 배열을 모을 리스트
    
    for i in range(len(data['text'])):
        conf = int(data['conf'][i])
        # 빈칸이나 노이즈가 아닌 실제 인식된 글자만 필터링
        if conf > 0:
            print(f" - 글자: '{data['text'][i]}' / 신뢰도: {conf}%")
            confidences.append(conf)
            
    print(f"\n신뢰도 배열: {confidences}")

    # 5. 결과 이미지를 화면에 띄우기 (사진 속 화면처럼)
    cv2.imshow('Clean Plate Result', clean_plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("번호판 영역을 찾지 못했습니다. 면적이나 비율을 조절해 보세요.")