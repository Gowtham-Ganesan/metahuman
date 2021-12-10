import os
import cv2
import keras
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras import callbacks

%matplotlib inline

DIRECTORY = 'Dataset'

CATEGORIES1 = ['Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)','Speed limit (70km/h)',
              'Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)','Speed limit (120km/h)','No passing',
              'No passing veh over 3.5 tons','Right-of-way at intersection','Priority road','Yield','Stop','No vehicles',
              'Veh > 3.5 tons prohibited','No entry','General caution','Dangerous curve left','Dangerous curve right','Double curve',
               'Bumpy road','Slippery road','Road narrows on the right','Road work','Traffic signals','Pedestrians',
               'Children crossing','Bicycles crossing','Beware of ice/snow','Wild animals crossing','End speed + passing limits',
               'Turn right ahead','Turn left ahead','Ahead only','Go straight or right','Go straight or left','Keep right',
               'Keep left','Roundabout mandatory','End of no passing','End no passing veh > 3.5 tons','Car','Buildings',
               'Forest', 'Mountain','Sea','Street','Vehicle']
CATEGORIES = [str(i) for i in range(0,50)]

data = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        label = CATEGORIES.index(category)
        arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        try:
            new_arr = cv2.resize(arr, (120, 120))
        except:
            pass
        data.append([new_arr, label])

random.shuffle(data)
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)
X = X/255
X = X.reshape(-1, 120, 120, 1)
X.shape

model = Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))

model.add(Dense(51, activation = 'softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=9, validation_split=0.1)

model.save("final1.h5")
