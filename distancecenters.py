#code snippet for finding local peaks off of disance map translated from
#original code written by Quentin Geissmann

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv2

objects = cv2.imread("/home/dylan/Marbles/Tests/MomentCenter/25Jul2014Opti_000001.jpg")
results = objects.copy()

objects = cv2.cvtColor(objects, cv2.COLOR_BGR2GRAY)

# THIS IS THE LINE TO BLUR THE IMAGE CF COMMENTS OF THIS POST
#objects = cv2.blur(objects,(3,3))
[junk,ThrObjects] = (cv2.threshold(objects, 80, 255,
    cv2.THRESH_BINARY_INV))
cv2.imwrite('ThrObjects.jpg',ThrObjects)
distance = cv2.distanceTransform(ThrObjects,2,5)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

#/* In order to find the local maxima, "distance"
#     * is subtracted from the result of the dilatation of
#     * "distance". All the peaks keep the save value */

peaks = cv2.dilate(distance, kernel, iterations = 3)
ThrObjects = cv2.dilate(ThrObjects , kernel, iterations = 3)

# Now all the peaks should be exactly 0
peaks = peaks - distance

#and the non-peaks 255
[junk,peaks] = cv2.threshold(peaks,0,255,cv2.THRESH_BINARY)
peaks = peaks.astype('uint8')

peaks = cv2.bitwise_xor(peaks,ThrObjects)
peaks = cv2.dilate(peaks, kernel, iterations = 1)

#/* In order to map the peaks, findContours() is used.
# * The results are stored in "contours" */

contours,hierarchy = (cv2.findContours(peaks,
    cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE))


if len(contours)>0:
    moms = []
    centers = []
    circles = []
    i = 0
    x = 0
    y = 0
    while i<len(contours):
        moms.append(cv2.moments(contours[i]))
        if moms[i]['m00'] > 0:
            center = (np.array([moms[i]['m10']/moms[i]['m00'],
                moms[i]['m01']/moms[i]['m00']]))
            centers.append(center)   
            y = center[0].astype(float)
            x = center[1].astype(float)
            rad = distance[x,y]
            circles.append(np.array([x,y,rad]))
            cv2.circle(results,(y.astype(int),x.astype(int)),rad.astype(int)+1,(255,0,0),2,4,0)
        i = i + 1
    cv2.imwrite("MomCenRes.jpg",results)
