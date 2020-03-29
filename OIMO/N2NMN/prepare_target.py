from resnet152 import ResNet152
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import time
import os
import glob

model = ResNet152(include_top=False, weights='imagenet')

intermediate_model = Model(inputs=model.input,outputs=model.layers[-3].output)

lines = open("../../demo_imgs.txt","r").readlines()

target = []

for i in range(len(imgs)):
        imgpath = lines[i].split("\n")[0]
	img = image.load_img(imgpath, target_size=(448, 448))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	pred = intermediate_model.predict(x)
        target.append(pred)

target = np.array(target)
np.save("final_targets.npy",target)
