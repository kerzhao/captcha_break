#encoding:UTF-8

import cv2
import os
import numpy as np
import random
from datetime import datetime
from generator import captcha_generator as gen
        
from keras.models import *
from keras.layers import *
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghijlmnqrtuwxy1234567890"
width, height, n_len, n_class = 140, 43, 6, len(chars)

#----------------------------------------------------------------------
def make_model(nb_conv, nb_pool, optimizer):
    """"""
    input_tensor = Input((height, width, 3))
    x = input_tensor
    
    x = Conv2D(32, (nb_conv, nb_conv), activation='relu')(x)
    x = Conv2D(32, (nb_conv, nb_conv), activation='relu')(x)
    x = MaxPooling2D((nb_pool, nb_pool))(x)
    
    x = Conv2D(64, (nb_conv, nb_conv), activation='relu')(x)
    x = Conv2D(64, (nb_conv, nb_conv), activation='relu')(x)
    x = MaxPooling2D((nb_pool, nb_pool))(x)
    
    x = Conv2D(128, (nb_conv, nb_conv), activation='relu')(x)
    x = Conv2D(128, (nb_conv, nb_conv), activation='relu')(x)
    x = MaxPooling2D((nb_pool, nb_pool))(x)
    
    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
    model = Model(inputs=input_tensor, outputs=x)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model

my_classifier = KerasClassifier(make_model, batch_size=32)
print 'create 60000 train samples'
print 'start at ', datetime.now()
X_train, y_train = gen(width, height, batch_size=60000)
print 'create 60000 train samples'
print 'end at ', datetime.now()
print 'create 10000 test samples'
print 'start at ', datetime.now()
X_test, y_test = gen(width, height, batch_size=10000)
print 'create 10000 test samples'
print 'end at ', datetime.now()

validator = GridSearchCV(my_classifier,
                         param_grid={# nb_epoch is avail for tuning even when not
                                     # an argument to model building function
                                     'nb_epoch': [3, 6],
                                     'nb_conv': [3],
                                     'nb_pool': [2],
                                     'optimizer': ['sgd', 
                                                   'rmsprop', 
                                                   'adagrad',
                                                   'adadelta',
                                                   'adam',
                                                   'adamax',
                                                   'nadam']},
                         scoring='neg_log_loss',
                         n_jobs=4)
validator.fit(X_train, y_train)

print('The parameters of the best model are: ')
print(validator.best_params_)

# validator.best_estimator_ returns sklearn-wrapped version of best model.
# validator.best_estimator_.model returns the (unwrapped) keras model
best_model = validator.best_estimator_.model
best_model.save('v3_best_model.h5')
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(X_test, y_test)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
    