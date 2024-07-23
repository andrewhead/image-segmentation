```bash
# First, install Python 3...
# Then install the requirements from `requirements.txt`
# (I'll skip describing what commands to run here because the last time I did
# Alyssa said they did not work for her).

# Then, download the Numpy masks file for the image and put it in 'data/'
# The masks I used for this are at:
# https://drive.google.com/file/d/1063Gdnbrq1jUvUYLh9vfMAkISDw24tAO/view?usp=sharing
# These were just generated using Alyssa's Image Segmenter Notebook.

# Then, after all of that, you can run...
python extract-segments.py \
  examples/sam-model.png \
  data/sam-model_masks.npy \
  --segment-output-dir segments \
  --traces-output-dir traces

# Look in the 'traces' output directory for examples of images where each of the
# segments have been individually 'traced' with a tight-fitting red border.
```
