from IPython import embed
import argparse
import os
from PIL import Image

import numpy as np
import cv2


parser = argparse.ArgumentParser("extract segments from image")
parser.add_argument("image")
parser.add_argument("masks_file")
parser.add_argument("--segment-output-dir")
parser.add_argument("--traces-output-dir")
parser.add_argument("--interactive", action="store_true")
args = parser.parse_args()


data = np.load(args.masks_file, allow_pickle=True)
data = data.item()

for segment_name in data.keys():
    # Much of this code is courtesy of ChatGPT
    image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2]

    # Add alpha channel (not sure if necessary; GPT suggested it)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GBRA)

    # Get segment mask
    segment = data[segment_name]
    mask = segment['segmentation']
    assert mask.shape == (height, width)

    # Make all parts of the image that weren't masked transparent.
    alpha_mask = np.where(mask == True, 255, 0).astype(np.uint8)
    masked_image = image.copy()
    masked_image[:, :, 3] = alpha_mask

    # Crop out the non-alpha pixels
    alpha_coords = cv2.findNonZero(masked_image[:, :, 3])
    x, y, w, h = cv2.boundingRect(alpha_coords)
    # print(x,y,w,h)  # print out bounding boxes
    cropped = masked_image[y:y+h, x:x+w]

    # Output the cropped file
    if args.segment_output_dir:
        if not os.path.exists(args.segment_output_dir):
            os.makedirs(args.segment_output_dir)
        cropped_file = os.path.join(args.segment_output_dir, segment_name + ".png") 
        cv2.imwrite(cropped_file, cropped)
        if args.interactive:
            Image.open(cropped_file).show()

    # Output the image with the segment traced on the edges
    if args.traces_output_dir:
        if not os.path.exists(args.traces_output_dir):
            os.makedirs(args.traces_output_dir)

        binary_mask = np.where(mask == True, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Grayscale the image to make the colorized pixels stand out.
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        highlighted_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(highlighted_image, contours, -1, (0, 0, 255, 255), 2)

        highlighted_file = os.path.join(args.traces_output_dir, segment_name + ".png")
        cv2.imwrite(highlighted_file, highlighted_image)

        if args.interactive:
            cv2.imshow("Contours", highlighted_image)
            cv2.waitKey(0)

