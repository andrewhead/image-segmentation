from IPython import embed
import argparse
import os

import numpy as np
import cv2


parser = argparse.ArgumentParser("extract segments from image")
parser.add_argument("image")
parser.add_argument("masks_file")
parser.add_argument("output_dir")
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
    image[:, :, 3] = alpha_mask

    # Crop out the non-alpha pixels
    alpha_coords = cv2.findNonZero(image[:, :, 3])
    x, y, w, h = cv2.boundingRect(alpha_coords)
    print(x,y,w,h)
    cropped = image[y:y+h, x:x+w]

    # Output the cropped file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cv2.imwrite(os.path.join(args.output_dir, segment_name + ".png"), cropped)

