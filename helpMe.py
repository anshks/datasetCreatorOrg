# import the necessary packages
import numpy as np
import cv2
import argparse
import os

# construct the argument parser and parse input image arguent
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image filename")
ap.add_argument("-b", "--background", required=True, help="Background filename")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
bg = cv2.imread(args["background"])

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

# applying edge detection we can find the outlines of objects in
# images
edged = auto_canny(gray)

cnt = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = cnt[1]
output = image.copy()

height = bg.shape[0]
width = bg.shape[1]

x,y,w,h = cv2.boundingRect(cnt)

x1=random.randrange(0, width-w)
y1=random.randrange(0, height-h)

bg[y1:y1+h, x1:x1+w] = output[y:y+h, x:x+w]

cv2.imshow("final", bg)
cv2.waitKey(0)

cv2.destroyAllWindows()
