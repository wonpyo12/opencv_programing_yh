import numpy as np
import cv2 as cv
 
# Create a black image
img = cv.imread("./my_id_card.png")
temp_img = img.copy()
mainimg = img.copy()
h,w = img.shape[:2] 
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
 
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,img,temp_img,mainimg
 
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        img = mainimg.copy()
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            temp_img = img.copy()
            if mode == True:
                cv.rectangle(temp_img,(ix,iy),(x,y),(0,255,0),2)
            
                cv.putText(temp_img, "FACE", (min(ix, x), min(iy, y) - 5), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
 
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img,(ix,iy),(x,y),(0,255,0),2)
            cv.putText(img, "FACE", (min(ix, x), min(iy, y) - 5), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        temp_img = img.copy()
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
 
while(1):
    cv.imshow('image',temp_img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):
        cv.imwrite("my_id_card_final.png",img)
    elif k == ord('q'):
    
        break
 
cv.destroyAllWindows()
