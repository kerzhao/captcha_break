import cv2
import os
import numpy as np
import random
from generator_v7 import captcha_generator as gen
        
from keras.models import *
from keras.layers import *
from keras import callbacks

chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghijlmnqrtuwxy"
width, height, n_len, n_class = 140, 44, 6, len(chars)

input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(3):
    x = Conv2D(32*2**i, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(6)]
model = Model(input=input_tensor, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

cbs = callbacks.TensorBoard(log_dir='./graph', 
                            histogram_freq=1, 
                            write_graph=True, 
                            write_images=True)

model.fit_generator(gen(width=width, height=height), steps_per_epoch=2000, epochs=10, 
                    validation_data=gen(width=width, height=height), validation_steps=500,
                    callbacks=[cbs])

model.save('mycnn_v201705020943_v7_adadelta.h5')
print 'saved mycnn_v201705020943_v7_adadelta.h5'

from tqdm import tqdm
def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen(width=width, height=height)
    for i in tqdm(range(batch_num)):
        X, y = generator.next()
        y_pred = model.predict(X)
        batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
    return batch_acc / batch_num

print evaluate(model)
