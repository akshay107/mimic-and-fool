import numpy as np
import os
import cv2
import keras
from keras.applications import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.datasets import mnist
import keras.backend as K
from keras.layers import ZeroPadding2D, Input
from keras.models import Model
from keras.engine.topology import Layer
from keras import initializers,regularizers
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.applications import InceptionV3
from keras.models import Model

# Learning phase is set to 0 since we want the network to use the pretrained moving mean/var
K.clear_session()
K.set_learning_phase(0)

model = InceptionV3()
interim_model = Model(inputs = model.input, outputs = model.layers[-2].output)
interim_model.load_weights('inception_v3_new.h5')

del(model)

class MyLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('glorot_uniform')
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape) == 4
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(299,299,3),
                                      initializer=self.init,
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        prog = K.tanh(self.W)
        out = x + prog
        return out
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3])


input_shape = (299,299,3)

x = Input(shape=input_shape)
x_aug = MyLayer()(x)
out = interim_model(x_aug)
new_model = Model(inputs=x,outputs=out)
new_model.layers[-1].trainable = False
new_model.summary()
adam = Adam(lr=0.0125)
new_model.compile(loss='mse', optimizer = adam)


targets = np.load("final_targets.npy")
lines = open("../../demo_imgs.txt","r").readlines()

img = cv2.imread('../COCO_train2014_000000000009.jpg')
img = cv2.resize(img,(299,299))
img = img[:,:,::-1]

img = img.astype('float32')
img /=255
img -= 0.5
img *= 2

img = np.arctanh(img*0.9999)

X_train = np.expand_dims(img,0)

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
        new_model.fit(X_train, targets[i:i+1], batch_size=1, epochs=300, verbose=1)
        W_aft = new_model.get_weights()[0]
        W_aft = np.tanh(W_aft)
        np.save("adv_imgs/"+fname,W_aft)
