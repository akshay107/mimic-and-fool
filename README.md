This repository contains code for the paper:

* Mimic and Fool: A Task Agnostic Adversarial Attack. Akshay Chaturvedi and Utpal Garain. IEEE TNNLS, 2020. [pdf](https://ieeexplore.ieee.org/document/9072347) 

## Dependencies

1. Keras v2.2.4 with Tensorflow (v 1.12.0) backend
3. numpy, opencv-python

The finetuned feature extractor weights can be downloaded from:

* Show and Tell: https://drive.google.com/open?id=1q3xJnp7HmFQWcquIvss1oUJJzawg73pY

* Show Attend and Tell: https://drive.google.com/open?id=1QmIze742K4aWTOuzhRF4nbiw5lHgN8TY

Extract the two files in their respective folders.

Under any setting, run prepare_target.py followed by main.py. The adversarial files will be saved in adv_imgs/ folder.
