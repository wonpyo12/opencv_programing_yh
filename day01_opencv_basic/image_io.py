import cv2 as cv
import sys
import urllib.request
import os
def get_samples(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url,filename)
    return cv.imread(filename)

img = get_samples("orange.jpg")
# img_gray = get_samples("./samples/starry_night.jpg",cv.IMREAD_GRAYSCALE)
img_gray=cv.imread("orange.jpg",cv.IMREAD_GRAYSCALE)
if img is None:
    sys.exit("Could not read the image.")
 
cv.imshow("Display window", img)
cv.imshow("Display window_gray",img_gray)
print("컬러이미지")
print(f"shape: {img.shape}")
print(f"dtype: {img.dtype}")
print(f"size: {img.size}")
print("그래이스케일");
print(f"shape: {img_gray.shape}")
print(f"dtype: {img_gray.dtype}")
print(f"size: {img_gray.size}")
k = cv.waitKey(0)
 
if k == ord("s"):
    cv.imwrite("starry_night.png", img)
    cv.imwrite("starry_night.png", img_gray)
 