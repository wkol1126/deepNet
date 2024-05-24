from common.util import im2col #image to columns
import numpy as np

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape #Filter number, channel, filter height, filter width    
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)  

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1)
        out = np.dot(col, col_W)

        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2) # (N,C,H,W)
        return out
    