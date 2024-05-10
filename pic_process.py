import cv2 as cv

input_path = 'pic/6.jpg'
dest_path = '6_28x28.jpg'
img = cv.imread(input_path,cv.IMREAD_GRAYSCALE)
h = 28
w = 28
img = cv.resize(img,(h,w),interpolation=cv.INTER_AREA)
print(img.shape)
#cv.imshow('pic', img)
cv.imwrite(dest_path, img)

img2 = cv.imread(dest_path)
#cv.imshow('img2', img2)
print(img2.shape)
cv.waitKey(0)
cv.destroyAllWindows()