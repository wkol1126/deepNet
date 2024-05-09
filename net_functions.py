import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    # c = np.max(a)
    # exp_a = np.exp(a-c) #防溢位
    # sum_exp_a = np.sum(exp_a)
    # y = exp_a / sum_exp_a
    # return y

def cross_entropy_error(y, t):
    # delta = 1e-7
    # out = -np.sum(t * np.log(y + delta))
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    # return out

def mnist_one_hot_coding(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[int(X[idx])] = 1 
    return T