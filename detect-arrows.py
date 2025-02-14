# Credit: https://stackoverflow.com/questions/66718462/how-to-detect-different-types-of-arrows-in-image
import cv2
import numpy as np
import argparse


def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode


def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])


parser = argparse.ArgumentParser(description="detect arrows in image")
parser.add_argument("image_file")
args = parser.parse_args()

img = cv2.imread(args.image_file)

contours, hierarchy = cv2.findContours(preprocess(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
    hull = cv2.convexHull(approx, returnPoints=False)
    sides = len(hull)

    if 6 > sides > 3 and sides + 2 == len(approx):
        arrow_tip = find_tip(approx[:,0,:], hull.squeeze())
        if arrow_tip:
            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
            cv2.circle(img, arrow_tip, 3, (0, 0, 255), cv2.FILLED)

cv2.imshow("Image", img)
cv2.waitKey(0)
