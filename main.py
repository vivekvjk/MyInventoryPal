
from skimage.filters import threshold_otsu, threshold_local
import numpy as np
import argparse
import imutils
import cv2
import sys
from imutils.perspective import four_point_transform

# arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = ap.parse_args()

#read and dispaly original image
image = cv2.imread(args.image)

ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("image",gray)

#blur and then canny

blurred = cv2.GaussianBlur(gray, (11,11),0)
edge = cv2.Canny(blurred, 50, 150)
cv2.imshow("edge",edge)
cv2.waitKey()



cnts, hiech = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

screenCnt = None
for c in cnts:
	perimeter = cv2.arcLength(c,True)
	epsilon = 0.1*cv2.arcLength(c,True)
	approx = cv2.approxPolyDP(c,epsilon,True)
	print(c)

	cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
	cv2.imshow('im', image)
	cv2.waitKey(0)
	print(len(approx))
	if (len(approx) == 4):
		screenCnt = approx
		break


warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)



