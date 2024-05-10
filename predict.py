from simple_net import TwoLayerNet
from multi_net import Multi_LayerNet
import pickle
import numpy as np
import cv2 as cv
from sklearn.datasets import fetch_openml

# X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
# i = 100
# X_img = np.uint8(X[i]).reshape(28,28)
# XX = X[i]/255
# XX = XX.reshape(1,-1)
# cv.imshow('mnist',X_img)

with open('weight.pkl', 'rb') as f:
    params = pickle.load(f)

model = Multi_LayerNet(784,50,10,params=params)

# print(params['b1'])
# print(model.layers['Affine1'].b)

path = '0_28x28.jpg'
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
img = cv.bitwise_not(img)
cv.imshow('img',img)
cv.waitKey(0)
img = img.ravel().reshape(1,-1)/255

#--------------------------------------
model.predict(img)
t = np.array([[0,1,0,0,0,0,0,0,0,0]])
model.loss(img, t)
#-------------------------------------
print(f'Predict : {(model.last_layer.y)}')
# print('label :', y[i])