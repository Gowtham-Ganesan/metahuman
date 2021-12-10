import numpy as np
from PIL import Image
import cv2
import tensorflow as tf


model_path = "Traffic.h5"
loaded_model = tf.keras.models.load_model(model_path)
cap = cv2.VideoCapture(0)
while True:
    ret, test_img = cap.read()
    filename = 'Temp/video.jpg'
    cv2.imwrite(filename, test_img)
    image = cv2.imread(filename)

    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((30, 30))
    expand_input = np.expand_dims(resize_image,axis=0)
    input_data = np.array(expand_input)
    input_data = input_data/255
    pred = loaded_model.predict(input_data)
    result = pred.argmax()
    CATEGORIES = ['Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)','Speed limit (70km/h)',
              'Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)','Speed limit (120km/h)','No passing',
              'No passing veh over 3.5 tons','Right-of-way at intersection','Priority road','Yield','Stop','No vehicles',
              'Veh > 3.5 tons prohibited','No entry','General caution','Dangerous curve left','Dangerous curve right','Double curve',
               'Bumpy road','Slippery road','Road narrows on the right','Road work','Traffic signals','Pedestrians',
               'Children crossing','Bicycles crossing','Beware of ice/snow','Wild animals crossing','End speed + passing limits',
               'Turn right ahead','Turn left ahead','Ahead only','Go straight or right','Go straight or left','Keep right',
               'Keep left','Roundabout mandatory','End of no passing','End no passing veh > 3.5 ton']
    print(CATEGORIES[result])
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
