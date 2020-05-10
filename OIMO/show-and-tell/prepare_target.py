import numpy as np
from keras.applications import InceptionV3
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import os

K.clear_session()
K.set_learning_phase(0)

model = InceptionV3()
interim_model = Model(inputs = model.input, outputs = model.layers[-2].output)

del(model)

interim_model.load_weights('inception_v3_new.h5')

sess = tf.Session()


lines = open("../../demo_imgs.txt","r").readlines()

target = []

for i in range(len(lines)):
	imgpath = lines[i].split("\n")[0]
	encoded_image = tf.gfile.GFile(imgpath,"rb").read()
	image = tf.image.decode_jpeg(encoded_image, channels=3)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image = tf.image.resize_images(image,size=[346,346],method=tf.image.ResizeMethod.BILINEAR)
	image = tf.image.resize_image_with_crop_or_pad(image, 299,299)
	image = tf.subtract(image, 0.5)
	image = tf.multiply(image, 2.0)
	img = sess.run(image)
	img = np.expand_dims(img,0)
	pred = interim_model.predict(img)
	target.append(pred)

target = np.array(target)
target = target.squeeze()
np.save("final_targets.npy",target)
