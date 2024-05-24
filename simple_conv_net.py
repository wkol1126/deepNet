import numpy as np
from collections import OrderedDict
from common.layers import Affine, Relu, Pooling
from common.layers import Convolution, SoftmaxWithLoss 

# input_dim 輸入資料 (channel, height, width)
# conv_params 捲積層超參數(字典)
    # filter_num 濾鏡數量
    # filter_size 濾鏡大小
    # stride 步幅
    # pad 填補
# hidden_size 隱藏層(全連接層)神經元數目
# output_size 輸出層(全連接層)神經元樹目
# weight_init_std 初始化權重的標準差

# Convolution initialize - 1st step


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_params = {'filter_num':30,
                                'filter_size':5,
                                'stride':1,
                                'pad':0},
                                hidden_layer_size=100,
                                output_layer_size=10,
                                weight_init_std = 0.01, params = None):
        filter_num = conv_params['filter_num']
        filter_size = conv_params['filter_size']
        filter_pad = conv_params['pad']
        filter_stride = conv_params['stride']
        input_size = input_dim[1]
        conv_output_size = int((input_size - filter_size + 2 * filter_pad)/filter_stride + 1)
        pool_output_size = int(filter_num *(conv_output_size/4) *
                               (conv_output_size/4)) 
        
        if not params:
            self.params = {}
        # Convolution layer
            self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size) 
            self.params['b1'] = weight_init_std * np.zeros(filter_num)
        # Affine layer
            self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_layer_size)
            self.params['b2'] = weight_init_std * np.zeros(hidden_layer_size)
            self.params['W3'] = weight_init_std * np.random.randn(hidden_layer_size, output_layer_size)
            self.params['b3'] = weight_init_std * np.zeros(output_layer_size)
        else:
            self.params = params 
#產生各層
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                           self.params['b1'],
                                           filter_stride,
                                           filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=4, pool_w=4, stride=4)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        
        self.last_layer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x    
    
    #x: 輸入資料, t:訓練資料
    def loss(self, x ,t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t):
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
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db   
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db   
        return grads

