import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../utils')
sys.path.append(dir_path + '/../tensorflow')


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
from caffe2.python import workspace

import keras_to_caffe2

def conv_model(l1=0.00000, l2=0.0001):
    model = Sequential([
                Conv2D(5, 5, activation='relu', name='conv1', 
                       data_format='channels_first', kernel_regularizer=regularizers.l2(l2),
                       input_shape=(3, 128, 128)),
                BatchNormalization(axis=1, name='batchnorm1'),
                MaxPooling2D(2, name='pool1', data_format='channels_first'),
                Conv2D(10, 5, activation='relu', name='conv2', 
                       data_format='channels_first', kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(axis=1, name='batchnorm2'),
                MaxPooling2D(2, name='pool2', data_format='channels_first'),
                Conv2D(20, 5, activation='relu', name='conv3',  
                       data_format='channels_first', kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(axis=1, name='batchnorm3'),
                MaxPooling2D(2, name='pool3', data_format='channels_first'),
                Conv2D(30, 5, activation='relu', name='conv4',  
                       data_format='channels_first', kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(axis=1, name='batchnorm4'),
                MaxPooling2D(2, name='pool4', data_format='channels_first'),
                Dropout(0.4, name='conv4_dropout'),
                Flatten(name='flatten'),
                Dense(300, activation='relu',    name='dense1', kernel_regularizer=regularizers.l2(l2)),
                Dropout(0.3, name='dense1_dropout'),
                Dense(300, activation='relu',    name='dense2', kernel_regularizer=regularizers.l2(l2)),
                Dropout(0.3, name='dense2_dropout'),
                Dense(9,   activation='softmax', name='out', kernel_regularizer=regularizers.l2(l2))
                ])
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':

    # Load Keras Model
    keras_model = conv_model()
    
    # Copy model from keras to caffe2
    caffe2_model = keras_to_caffe2.keras_to_caffe2(keras_model)
    
    # Generate random input data
    frame = np.random.random_sample((1, 3, 128, 128)).astype(np.float32)
    
    # Predict using Keras 
    keras_pred = keras_model.predict(frame)[0]
    
    # Predict using Caffe2 
    workspace.FeedBlob('in', frame)
    workspace.RunNet(caffe2_model.net)
    caffe2_pred = workspace.FetchBlob('softmax')[0]
    
    # Compare Predictions
    print(keras_pred.tolist())
    print(caffe2_pred.tolist())
    print('%d == %d' % (np.argmax(keras_pred), np.argmax(caffe2_pred)))
