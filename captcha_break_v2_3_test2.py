import cv2
import os
import numpy as np
import random
#from generator import captcha_generator as gen
from captcha.image import ImageCaptcha
        
from keras.models import *
from keras.layers import *

chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
width, height, n_len, n_class = 250, 80, 6, len(chars)

def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(6)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(4):
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(6)]
model = Model(input=input_tensor, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


model.fit_generator(gen(), steps_per_epoch=2000, epochs=20, 
                    validation_data=gen(), validation_steps=500)

model.save('mycnn_v20170424_adagrad.h5')
print 'saved mycnn_v20170424_adagrad.h5'

from tqdm import tqdm
def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen(1)
    for i in tqdm(range(batch_num)):
        X, y = generator.next()
        y_pred = model.predict(X)
        batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
    return batch_acc / batch_num

evaluate(model)
