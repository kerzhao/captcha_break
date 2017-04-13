import cv2
import os
import numpy as np
import random

import string
characters = string.ascii_uppercase
undistinct = 'CKOPSVZ'
chars = characters
for i in characters:
    if i not in undistinct:
        chars += i.lower()
print(chars)

width, height, n_len, n_class = 130, 53, 4, len(chars)
path = os.getcwd() + '/sample3'
print path

root, dirs, files = os.walk(path).next()

def getAllImages(path):
    cnt = len(files)
    X = np.zeros((cnt, height, width, 1), dtype=np.uint8)
    y = []
    for i, j in enumerate(files):
        #X[i] = cv2.imread(root+'/'+j)
        img = cv2.imread(root+'/'+j)
        blur = cv2.bilateralFilter(img, 9, 75, 75)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X[i] = grayimg.reshape((height, width, 1))
        y.append(j[:4])
    return X, y

allX, ally = getAllImages(path)

def gen(batch_size=8):
    X = np.zeros((batch_size, height, width, 1), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    cnt = 0
    while True:
        for i in range(batch_size):
            X[i] = allX[cnt]
            for j, ch in enumerate(ally[cnt]):
                y[j][i, :] = 0
                y[j][i, chars.find(ch)] = 1
            if cnt >= len(files) - 1:
                cnt = 0
            else:
                cnt += 1
        yield X, y
        
from keras.models import *
from keras.layers import *

input_tensor = Input((height, width, 1))
x = input_tensor
#for i in range(4):
    #x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    #x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    #x = MaxPooling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.05)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
model = Model(inputs=input_tensor, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit_generator(gen(), steps_per_epoch=2000, epochs=200,
                    validation_data=gen(), validation_steps=30)

model.save('mycnn.h5')

from tqdm import tqdm
def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen()
    for i in tqdm(range(batch_num)):
        X, y = generator.next()
        y_pred = model.predict(X)
        batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
    return batch_acc / batch_num

evaluate(model)