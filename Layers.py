import numpy as np

class AddLayer:
    def __init__(self) -> None:
        pass

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy 

class MultiLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = self.x * self.y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
    

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x<0)
        out = x.copy()
        return out[self.mask] = 0

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout * 1
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 +  np.exp(x)) 
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx        

class Affine:
     def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

     def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
     
     def backward(self, dout):
         dx = np.dot(dout, np.transpose(self.W))
         self.dW = np.dot(np.transpose(self.x), dout)
         self.db = np.sum(dout, axis=0)



def main1():
    apple = 100
    apple_num = 2
    tax = 1.1

    #layer
    multi_apple_layer = MultiLayer()
    multi_tax_layer = MultiLayer()

    #forward
    apple_price = multi_apple_layer.forward(apple, apple_num) 
    price = multi_tax_layer.forward(apple_price, tax)

    #backward
    dout = 1
    dapple_price, dtax = multi_tax_layer.backward(dout)
    dapple, dapple_num = multi_apple_layer.backward(dapple_price)

    print('{},{},{}'.format(dapple, dapple_num, dtax))    

def main2():
    apple_num = 2
    apple = 100
    orange_num = 3
    orange = 150
    tax = 1.1

    #layers
    multi_apple_layer = MultiLayer()
    multi_orange_layer = MultiLayer()
    add_apple_orange_layer = AddLayer()
    multi_tax_layer = MultiLayer()

    #forward
    apple_price = multi_apple_layer.forward(apple, apple_num)
    orange_price = multi_orange_layer.forward(orange, orange_num)
    apple_orange_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price_with_tax = multi_tax_layer.forward(apple_orange_price, tax)

    #backward
    dout = 1
    dapple_orange_price, dtax = multi_tax_layer.backward(dout)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dapple_orange_price)
    dapple, dapple_num = multi_apple_layer.backward(dapple_price)
    dorange, dorange_num = multi_orange_layer.backward(dorange_price)

    print('Total proce:', price_with_tax)
    print(dapple, dapple_num, dorange, dorange_num, dtax)


if __name__ == '__main__':
    main2()
    