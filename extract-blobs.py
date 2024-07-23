from IPython import embed
import argparse
import os
from PIL import Image

import numpy as np
import cv2


parser = argparse.ArgumentParser("extract blobs from image")
parser.add_argument("image")
parser.add_argument("--blob-output-dir")
parser.add_argument("--traces-output-dir")
parser.add_argument("--show-boxes", action="store_true")
parser.add_argument("--interactive", action="store_true")
args = parser.parse_args()


# Much of this code is (still) courtesy of ChatGPT
# ... and some of it is from unattributed areas of Stack Overflow :grimaces:
image = cv2.imread(args.image)
# padded = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT,value=[255,255,255])
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
thresh_inverted = cv2.bitwise_not(thresh)
contours, _ = cv2.findContours(thresh_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for i, contour in enumerate(contours):
    with_contours = cv2.drawContours(image.copy(), [contour], 0, (0, 0, 255), 2)

    if args.blob_output_dir:
        if not os.path.exists(args.blob_output_dir):
            os.makedirs(args.blob_output_dir)
        path = os.path.join(args.blob_output_dir, str(i) + ".png")
        cv2.imwrite(path, with_contours)

        if args.interactive:
            cv2.imshow("Blobs", with_contours)
            cv2.waitKey(0)

    if args.traces_output_dir:
        alpha_ok = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        alpha = np.zeros_like(mask)
        alpha_ok[:, :, 3] = mask
 
        # Crop out the non-alpha pixels
        alpha_coords = cv2.findNonZero(alpha_ok[:, :, 3])
        x, y, w, h = cv2.boundingRect(alpha_coords)
        cropped = alpha_ok[y:y+h, x:x+w]

        # Output the cropped file
        if not os.path.exists(args.traces_output_dir):
            os.makedirs(args.traces_output_dir)
        cropped_file = os.path.join(args.traces_output_dir, str(i) + ".png") 
        cv2.imwrite(cropped_file, cropped)
        if args.interactive:
            Image.open(cropped_file).show()


if args.show_boxes:
    with_boxes = image.copy()
    # Drawing code inspired by https://stackoverflow.com/a/40204315/2096369
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(with_boxes, (x,y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Boxes", with_boxes)
    cv2.waitKey(0)

