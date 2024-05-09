import numpy as np
from collections import OrderedDict
from Layers import *
from common.layers import *
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):

        #權重初始化
        self.params = {} # 建立參數字典
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = weight_init_std * \
                            np.random.randn(hidden_size)

        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = weight_init_std * \
                            np.random.randn(output_size)
        
        #產生各層
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        self.last_layer = Softmax_with_loss()
        #self.last_layer = SoftmaxWithLoss()
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x    
    
    #x: 輸入資料, t:訓練資料
    def loss(self, x ,t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def acuuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy
    
    def gradient(self, x, t):

        #forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
#        print('1----->',dout[1])
        for layer in layers:
#            print('2---->',layer)
            dout = layer.backward(dout)
#            print('2out---->',dout)
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db   

        return grads