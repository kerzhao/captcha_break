import cv2
import os
import numpy as np
import random
from generator import captcha_generator as gen
        
from keras.models import *
from keras.layers import *

chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghijlmnqrtuwxy1234567890"
width, height, n_len, n_class = 140, 44, 6, len(chars)

input_tensor = Input((height, width, 3))
x = input_tensor
#for i in range(4):
    #x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    #x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    #x = MaxPooling2D((2, 2))(x)

#x = Conv2D(32, (5, 5), activation='relu')(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#x = Conv2D(64, (5, 5), activation='relu')(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#x = Conv2D(128, (5, 5), activation='relu')(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#x = Conv2D(256, (3, 3), activation='relu')(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
model = Model(inputs=input_tensor, outputs=x)

#sgd = optimizers.SGD(lr=0.1, decay=1e-2, momentum=0.9)

model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

model.fit_generator(gen(width=width, height=height), steps_per_epoch=2000, epochs=20, 
                    validation_data=gen(width=width, height=height), validation_steps=500)

model.save('mycnn_v20170424_adagrad.h5')
print 'saved mycnn_v20170424_adagrad.h5'

from tqdm import tqdm
def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen(width=width, height=height)
    for i in tqdm(range(batch_num)):
        X, y = generator.next()
        y_pred = model.predict(X)
        batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
    return batch_acc / batch_num

evaluate(model)
