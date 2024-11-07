import cv2 as cv

#RAW Image
img = cv.imread('mamografi-ankara.jpg')
cv.imshow('Mamografi Image', img)

#To Gray
img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Mamografi Gri', img_gray)

#Resize
height,weight=img.shape[:2]
img_resize=cv.resize(img,(weight//(2),height))
cv.imshow('Resized', img_resize)

#Rotate
def rotate(img,angle, rotPoint=None):
    (H,W) = img.shape[:2]
    if(rotPoint==None):
        rotPoint = (W//2,H//2)
    rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions=(W,H)
    return cv.warpAffine(img,rotMat,dimensions)
angle=0
while(angle<361):
    cv.imshow("Topac",rotate(img,angle))
    cv.waitKey(10)
    angle=angle+1
    if(angle==360):
        angle=0

## Görüntü Filtreleme

# Gaussian Blur
img2=cv.GaussianBlur(img,(3,3),0)
img2_gray=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
cv.imshow('Gaussian',img2)

#bilateral Filter
img3=cv.bilateralFilter(img,-1,100,50)
img3_gray=cv.cvtColor(img3,cv.COLOR_BGR2GRAY)
cv.imshow('Bilateral',img3)


# Median BLur
img4 = cv.medianBlur(img, 5)
img4_gray=cv.cvtColor(img4,cv.COLOR_BGR2GRAY)
cv.imshow('Median',img4)


# Laplacian Filter
img7=cv.Laplacian(img_gray, cv.CV_64F )
img7=np.uint8(np.absolute(img7))
cv.imshow('Laplacian', img7)

img8_x= cv.Sobel(img_gray, cv.CV_64F, 0,1)
img8_y= cv.Sobel(img_gray, cv.CV_64F, 1,0)
img8= cv.bitwise_not(img8_x,img8_y)
cv.imshow('Sobel', img8)

# img9=cv.Canny(img_gray,125,175)
# cv.imshow('Canny',img9)
#
# img10=cv.Canny(img2_gray,125,175)
# cv.imshow('Canny with Gaussian',img10)
#
# img11=cv.Canny(img4_gray,125,175)
# cv.imshow('Canny with Median',img11)
#
# img12=cv.Canny(img3_gray,125,175)
# cv.imshow('Canny with Bilateral',img12)




clahe=cv.createCLAHE(clipLimit=5.0,tileGridSize=(3,3))
img5=clahe.apply(img_gray)
cv.imshow('Kontrast Arttırılmış Görüntü',img5)

img6 = cv.equalizeHist(img_gray)
cv.imshow('Histogram Eşitlemesi', img6)

cv.waitKey(0)



