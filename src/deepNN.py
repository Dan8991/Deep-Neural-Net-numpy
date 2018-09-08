import numpy as np
from keras.datasets import mnist

def init_params(layers_shape):
    W = []
    b = []
    for i in range(len(layers_shape) - 1):
        W.append(np.random.randn(layers_shape[i + 1], layers_shape[i]) * 0.01)
        b.append(np.zeros([layers_shape[i+1], 1]))
    return W,b

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def normalize_inputs(x):
    return x/256-0.5

def one_hot(y):
    res = np.zeros([10, y.shape[1]])
    for i in range(y.shape[1]):
        res[y[0, i], i] = 1
    return res

def forward_prop(inputs, parameters):
    W1, W2, W3 = parameters["W"]
    b1, b2, b3 = parameters["b"]
    
    Z1 = np.dot(W1, inputs) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) +b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)
    cache = {"Z1":Z1, "Z2":Z2, "Z3":Z3, "A1":A1, "A2":A2, "A3":A3}
    return A3, cache

def loss(y_pred, y):
    m = y.shape[1]
    return -1/m*np.sum(np.sum(y*np.log(y_pred),axis=1), axis=0)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

m, h ,w = x_train.shape
x_train = normalize_inputs(np.reshape(x_train, [m, h*w]).T)
y_train = one_hot(np.reshape(y_train,[1,m]))

W, b = init_params([h*w, 128, 128, 10])
params = {"W": W, "b": b}
y_pred, cache = forward_prop(x_train, params)
print(loss(y_pred, y_train))