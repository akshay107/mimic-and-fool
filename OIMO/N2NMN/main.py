from resnet152 import ResNet152
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from tensorflow import convert_to_tensor
from keras.engine.topology import Layer
from keras import initializers,regularizers
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.layers import ZeroPadding2D, Input,Reshape, Lambda
import numpy as np
import time
import os
import glob
import tensorflow as tf
import keras.backend as K

# Learning phase is set to 0 since we want the network to use the pretrained moving mean/var
K.clear_session()
K.set_learning_phase(0)

model = ResNet152(include_top=False, weights='imagenet')

intermediate_model = Model(inputs=model.input,outputs=model.layers[-3].output)

mean = np.array([103.939, 116.779, 123.68]).astype('float32')
expanded_mean = np.tile(mean,(448,448,1))
expanded_mean = np.expand_dims(expanded_mean,0)
tf_mean = convert_to_tensor(expanded_mean)

class MyLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.scale = 50
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape) == 4
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(448,448,3),
                                      initializer=self.init,
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        out = x + self.scale*self.W
        return out
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3])


input_shape = (448,448,3)

clip_layer1 = Lambda(lambda xin: tf.where(xin+tf_mean>255.0,255.0 - tf_mean,xin),output_shape=(448,448,3))
clip_layer2 = Lambda(lambda xin: tf.where(xin+tf_mean<0.0, -tf_mean ,xin),output_shape=(448,448,3))

x = Input(shape=input_shape)
x_aug = MyLayer()(x)
x_aug = clip_layer1(x_aug)
x_aug = clip_layer2(x_aug)
out = intermediate_model(x_aug)

new_model = Model(inputs=x,outputs=out)

new_model.layers[-1].trainable = False

new_model.summary()

adam = Adam(lr=0.00625)
new_model.compile(loss='mse', optimizer = adam)

adv_img_model = Model(inputs = new_model.input, outputs = new_model.layers[-2].output)

targets = np.load("final_targets.npy")
lines = open("../demo_imgs.txt","r").readlines()

img_path = '../COCO_train2014_000000000009.jpg'
img = image.load_img(img_path, target_size=(448, 448))
X_train = image.img_to_array(img)
X_train = np.expand_dims(X_train, axis=0)
X_train = preprocess_input(X_train)

if not os.path.exists("adv_imgs"):
        os.makedirs("adv_imgs")

for i in range(len(lines)):
	print(i)
        K.set_value(new_model.optimizer.iterations,0)
        symbolic_weights = getattr(new_model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        if i>=1:
               for k in range(1,3):
                        weight_values[k][:] = 0
               new_model.optimizer.set_weights(weight_values)
        symbolic_weights = getattr(new_model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        fname = lines[i].split('\n')[0].split('/')[-1].replace('jpg','npy')
        targets = np.load("/home/cvpr/akshay/resnet_res5c/"+fname) 
	new_model.load_weights('init_weights.h5')
	new_model.fit(X_train, targets[i:i+1], batch_size=1, epochs=500, verbose=1)
	adv_img = adv_img_model.predict(X_train)
	img = adv_img + mean
        np.save("adv_imgs/"+fname,img)
