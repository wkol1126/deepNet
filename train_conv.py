from sklearn.datasets import fetch_openml
from net_functions import mnist_one_hot_coding
from sklearn.model_selection import train_test_split
from simple_conv_net import SimpleConvNet
import pandas as pd
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import numpy as np
from common.util import im2col
from common.optimizer import SGD, Adam
from dataset.mnist import load_mnist

# (X_train, y_train), (X_test, y_test) = load_mnist(flatten=False)
# print(X_train.shape, y_train.shape)

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')

print(X.shape, y.shape)
y_ohc = mnist_one_hot_coding(y)
X = X / 255 #normalize
X_train, X_test, y_train, y_test = train_test_split(X, y_ohc, train_size=0.8, random_state=42) 


X_train = X_train.reshape(-1,1,28,28)
X_test = X_test.reshape(-1,1,28,28)

print(X_train[0])




# 參數設定
train_size = X_train.shape[0]
test_size = X_test.shape[0]
batch_size = 100
iter_num = int(train_size / batch_size)

input_size = X_train.shape[2]
output_size = 10


learning_rate =0.01
# optimizer = SGD(learning_rate)
optimizer = Adam(learning_rate)
train_loss = []
train_acc = []
test_acc = []
iter_per_epoch = max(train_size/batch_size, 1)
# model = TwoLayerNet(input_size, hidden_size, output_size,weight_init)
model = SimpleConvNet()
epoch_num = 2
print('----Network information-----')
print('Data shape', X_train.shape, y_train.shape)
print('input size=',input_size )
print('output size=',output_size)
print(model.layers.keys())


    
epoch = 0
index = 0
print('Traning Start...')
for i in range(iter_num*epoch_num):
    
    #隨機梯度下降法SGD
    
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]
   
        
    grads = model.gradient(X_batch, y_batch)
    params = model.params
    #更新權重
    optimizer.update(params, grads )
    # for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
    #     model.params[key] -= learning_rate * grads[key]

    loss = model.loss(X_batch, y_batch)
    train_loss.append(loss)
        
    if i % iter_num == 0:
        epoch += 1

    if i % 10 == 0:
        index += 1
        print('epoch {} / {}'.format(epoch, epoch_num))   
        sample_mask_train = np.random.choice(train_size, 1000)
        sample_mask_test = np.random.choice(test_size, 1000)
        X_train_smaple = X_train[sample_mask_train]
        y_train_sample = y_train[sample_mask_train]
        X_test_sample = X_test[sample_mask_test]
        y_test_sample = y_test[sample_mask_test]
        train_acc.append(model.accuracy(X_train_smaple, y_train_sample))
        test_acc.append(model.accuracy(X_test_sample, y_test_sample))
        print('epoch {} / {}'.format(epoch, epoch_num))
        print(f'loss = {train_loss[i]:.3f}',end='')
        print(f'tran_acc = {np.sum(train_acc)/index:.3f}   test_acc= {np.sum(test_acc)/index:.3f}')
            
plt.plot(train_loss)        
plt.show()

#save weight parameter

import pickle
with open('weight_conv.pkl', 'wb') as f:
    pickle.dump(model.params, f)