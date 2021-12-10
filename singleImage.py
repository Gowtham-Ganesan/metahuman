import os
import cv2
from keras.models import load_model
import numpy as np

model = load_model('imp.h5')

def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (120, 120))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 120, 120, 1)
    return new_arr

img= sys.args[1]
CATEGORIES = ['Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)','Speed limit (70km/h)',
              'Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)','Speed limit (120km/h)','No passing',
              'No passing veh over 3.5 tons','Right-of-way at intersection','Priority road','Yield','Stop','No vehicles',
              'Veh > 3.5 tons prohibited','No entry','General caution','Dangerous curve left','Dangerous curve right','Double curve',
               'Bumpy road','Slippery road','Road narrows on the right','Road work','Traffic signals','Pedestrians',
               'Children crossing','Bicycles crossing','Beware of ice/snow','Wild animals crossing','End speed + passing limits',
               'Turn right ahead','Turn left ahead','Ahead only','Go straight or right','Go straight or left','Keep right',
               'Keep left','Roundabout mandatory','End of no passing','End no passing veh > 3.5 tons','Car','Buildings',
               'Forest','Glacier', 'Mountain','Sea','Street','Vehicle']
im = cv2.imread(img)
prediction = model.predict([image(img)])
print(CATEGORIES[prediction.argmax()])
