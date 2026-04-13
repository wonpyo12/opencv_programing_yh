import cv2
from pyzbar import pyzbar

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[오류] 웹캠을 열 수 없습니다.")
    exit()

print("QR Code Scanner 시작 - Step 1: 기본")
print("  q: 종료")


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    qr_codes = pyzbar.decode(frame)

    for qr in qr_codes:
        if qr.type != 'QRCODE':
            continue

        x, y, w, h = qr.rect

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        data = qr.data.decode('utf-8')

        cv2.putText(frame, data, (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

        print(f"[감지 QR] {data}")
        pass

    cv2.imshow('QR Code SCanner', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n[완료] Step 1 종료")