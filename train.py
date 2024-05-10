from sklearn.datasets import fetch_openml
from net_functions import mnist_one_hot_coding
from sklearn.model_selection import train_test_split
from simple_net import TwoLayerNet
from multi_net import Multi_LayerNet
import numpy as np
import pandas as pd
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
y_ohc = mnist_one_hot_coding(y)
X = X / 255
X_train, X_test, y_train, y_test = train_test_split(X, y_ohc, train_size=0.8, random_state=42) 



# (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, one_hot_label=True)


# 參數設定
iter_num = 10000
batch_size = 100
input_size = X_train.shape[1]

hidden_size = 50
output_size = y_train.shape[1]
train_size = X_train.shape[0]
batch_size = 50
learning_rate =0.1
weight_init = 2/np.sqrt(input_size)
train_loss = []
train_acc = []
test_acc = []
iter_per_epoch = max(train_size/batch_size, 1)
# model = TwoLayerNet(input_size, hidden_size, output_size,weight_init)
model = Multi_LayerNet(input_size, hidden_size, output_size,weight_init)
epoch = 0
print('----Network information-----')
print('Data shape', X_train.shape, y_train.shape)
print('input size=',input_size )
print('output size=',output_size)
print(model.layers.keys())

for i in range(iter_num):
    #隨機梯度下降法SGD
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]
    # X_batch = X_train[i:i+batch_size]
    # y_batch = y_train[i:i+batch_size]
    grads = model.gradient(X_batch, y_batch)

    #更新權重
    for key in ('W1', 'b1', 'W2', 'b2'):
        model.params[key] -= learning_rate * grads[key]

    loss = model.loss(X_batch, y_batch)
    train_loss.append(loss)
    
    
    if i % iter_per_epoch == 0:
        epoch += 1
        print('Traning Start...')
        print('epoch {} / {}'.format(epoch, int(iter_num // iter_per_epoch) + 1 ))
        train_acc.append(model.acuuracy(X_train, y_train))
        test_acc.append(model.acuuracy(X_test, y_test))
        print('loss = ',train_loss[i])
        print('tran_acc =',train_acc[epoch-1], 'test_acc=', test_acc[epoch-1])     
#plt.plot(train_loss)        
#plt.show()

#save weight parameter

import pickle
with open('weight.pkl', 'wb') as f:
    pickle.dump(model.params, f)

