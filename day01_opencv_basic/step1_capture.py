import numpy as np
import cv2 as cv
 
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
count =0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 1)#좌우 반전
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', frame)
    Key = cv.waitKey(1)
    if Key == ord('q'):
        break
    
    elif Key == ord('c'):
        filename = f"capture_{count}.jpg"
        cv.imwrite(filename,frame)
        print("캡처완료")
        count =count+1
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()