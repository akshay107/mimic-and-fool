import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import Model
from keras.applications import VGG16

mean_file = "./ilsvrc_2012_mean.npy"
mean = np.load(mean_file).mean(1).mean(1)

lines = open("../../demo_imgs.txt","r").readlines()

model = VGG16()

interim_model = Model(inputs=model.input,outputs=model.layers[-6].output)

del(model)
interim_model.load_weights('vgg_conv5.h5')

feat = []
for i in range(len(lines)):
	image_file = lines[i].split("\n")[0]
	image = cv2.imread(image_file)
	temp = image.swapaxes(0, 2)
	temp = temp[::-1]
	image = temp.swapaxes(0, 2)
	image = cv2.resize(image, (224,224))
	image = image - mean
	image = image.astype(np.float32)
	image = np.expand_dims(image,0)
	pred = interim_model.predict(image)
	pred = np.reshape(pred,(196,512))
	feat.append(pred)

feat_np = np.array(feat)
feat_np = feat_np.squeeze()
np.save("final_targets.npy",feat_np)
