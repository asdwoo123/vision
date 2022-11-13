import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50
from keras.preprocessing import image
import cv2 as cv
import numpy as np


IMAGE_SIZE = 224
base_model = ResNet50(weights='imagenet', include_top=False, input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
base_model.trainable = False
model = keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

def img_predict(imgPath):
    img = cv.imread(imgPath)
    img = cv.resize(img, None, fx=0.3, fy=0.3, interpolation=cv.INTER_LINEAR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 4)
    contours, hierarchy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    binary = cv.cvtColor(binary, cv.COLOR_GRAY2RGB)
    roi = (0, 0, 0, 0)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w > roi[2] and w < 300:
            roi = (x, y, w, h)
    x, y, w, h = roi
    img = img[y:y+h, x:x+w]
    img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE), None, None, interpolation=cv.INTER_LINEAR)
    img = np.expand_dims(img, axis=0)
    predict = model.predict(img)
    return predict

print(img_predict('images/ok1.jpg'))
print(img_predict('images/ok2.jpg'))



