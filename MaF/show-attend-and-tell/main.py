import numpy as np
import os
import keras
import keras.backend as K
from tensorflow import convert_to_tensor
import tensorflow as tf
from keras.models import Model
from keras.applications import VGG16
from keras.layers import ZeroPadding2D, Input,Reshape, Lambda
from keras.models import Model
from keras.engine.topology import Layer
from keras import initializers,regularizers
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

# Learning phase is set to 0 since we want the network to use the pretrained moving mean/var
K.clear_session()
K.set_learning_phase(0)

model = VGG16()
interim_model = Model(inputs=model.input,outputs=model.layers[-6].output)

del(model)

interim_model.load_weights('vgg_conv5.h5')

mean_file = "./ilsvrc_2012_mean.npy"
mean = np.load(mean_file).mean(1).mean(1)
mean = mean.astype('float32')
expanded_mean = np.tile(mean,(224,224,1))
expanded_mean = np.expand_dims(expanded_mean,0)
tf_mean = convert_to_tensor(expanded_mean)

class MyLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.scale = 125
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape) == 4
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(224,224,3),
                                      initializer=self.init,
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        out = x + self.scale*self.W
        return out
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3])


input_shape = (224,224,3)

clip_layer1 = Lambda(lambda xin: tf.where(xin+tf_mean>255.0,255.0 - tf_mean,xin),output_shape=(224,224,3))
clip_layer2 = Lambda(lambda xin: tf.where(xin+tf_mean<0.0, -tf_mean ,xin),output_shape=(224,224,3))

x = Input(shape=input_shape)
x_aug = MyLayer()(x)
x_aug = clip_layer1(x_aug)
x_aug = clip_layer2(x_aug)
x_aug = interim_model(x_aug)
out = Reshape((196,512))(x_aug)

new_model = Model(inputs=x,outputs=out)

new_model.layers[-2].trainable = False

new_model.summary()

adam = Adam(lr=0.025)
new_model.compile(loss='mse', optimizer = adam)

adv_img_model = Model(inputs = new_model.input, outputs = new_model.layers[-3].output)

targets = np.load("final_targets.npy")
lines = open("../../demo_imgs.txt","r").readlines()

X_train = np.zeros((1,224,224,3))

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
	new_model.fit(X_train, targets[i:i+1], batch_size=1, epochs=1000, verbose=1)
	adv_img = adv_img_model.predict(X_train)
	img = adv_img + mean
	pred = new_model.predict(X_train)
        np.save("adv_imgs/"+fname,img)
