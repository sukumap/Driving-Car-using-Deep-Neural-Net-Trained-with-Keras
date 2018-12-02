import csv
import cv2
import numpy as np
import os

lines = []
with open('../training3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines,test_size=0.2)
import sklearn.utils

def generator(samples,batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batchsamples = samples[offset:offset+batch_size]
            images =[]
            angles =[]
            steering_offset = 0.35
            for batch_sample in batchsamples:
                camera = np.random.choice(['center','left','right'])
                if camera == 'left':
                    name = '../training3/IMG/'+batch_sample[1].split('/')[-1]
                    center_angle = float(batch_sample[3])
                    center_angle += steering_offset
                elif camera == 'right':
                    name = '../training3/IMG/'+batch_sample[2].split('/')[-1]
                    center_angle = float(batch_sample[3])
                    center_angle  -= steering_offset
                else:
                    name = '../training3/IMG/'+batch_sample[0].split('/')[-1]
                    center_angle = float(batch_sample[3])

                flip_prob = np.random.random()

                center_image = cv2.imread(name)
                if flip_prob > 0.5:
                    center_image = cv2.flip(center_image,1)
                    center_angle = -1*center_angle
                images.append(center_image)
                angles.append(center_angle)
#                images.append(cv2.flip(center_image,1))
#                angles.append(center_angle*-1.0)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
#ch, row, col = 3, 80,320

train_generator = generator(train_samples,batch_size=32)
validation_generator = generator(validation_samples,batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Cropping2D, Dropout
from keras.layers import Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

dropout_rate = 1.0
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(dropout_rate))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(dropout_rate))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(dropout_rate))

#model.add(Convolution2D(25,5,5,activation="relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(100))

model.add(Dense(50))

model.add(Dense(10))
model.add(Dense(1))

model.compile(loss ='mse', optimizer='adam')
#model.fit(X_train,y_train, validation_split = 0.2,shuffle=True,nb_epoch=7)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),\
                    validation_data=validation_generator,\
                    nb_val_samples = len(validation_samples), nb_epoch=15)

model.save('model.h5')

