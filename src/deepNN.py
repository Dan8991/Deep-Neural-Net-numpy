import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


def init_params(layers_shape):
    W = []
    b = []
    for i in range(len(layers_shape) - 1):
        W.append(np.random.randn(layers_shape[i + 1], layers_shape[i]) * 0.01)
        b.append(np.zeros([layers_shape[i+1], 1]))
    return {"W": W, "b": b}


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


def calculate_gradients(X, Y, params, cache, mom_grads, beta=0.9):
    Z1, Z2, Z3 = cache["Z"]
    A1, A2, A3 = cache["A"]
    _, W2, W3 = params["W"]
    m = Y.shape[1]
    VdW1, VdW2, VdW3 = mom_grads["dW"]
    Vdb1, Vdb2, Vdb3 = mom_grads["db"]
    dZ3 = A3 - Y
    dW3 = 1/m*np.dot(dZ3, A2.T)
    db3 = 1/m*np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = np.dot(W3.T, dZ3)*(A2 > 0)
    dW2 = 1/m*np.dot(dZ2, A1.T)
    db2 = 1/m*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2)*(A1 > 0)
    dW1 = 1/m*np.dot(dZ1, X.T)
    db1 = 1/m*np.sum(dZ1, axis=1, keepdims=True)
    VdW1 = beta*VdW1 + (1 - beta)*dW1
    VdW2 = beta*VdW2 + (1 - beta)*dW2
    VdW3 = beta*VdW3 + (1 - beta)*dW3
    Vdb1 = beta*Vdb1 + (1 - beta)*db1
    Vdb2 = beta*Vdb2 + (1 - beta)*db2
    Vdb3 = beta*Vdb3 + (1 - beta)*db3
    dW = [VdW1, VdW2, VdW3]
    db = [Vdb1, Vdb2, Vdb3]
    grads = {"dW": dW, "db": db}

    return grads


def forward_prop(inputs, parameters):
    W1, W2, W3 = parameters["W"]
    b1, b2, b3 = parameters["b"]

    Z1 = np.dot(W1, inputs) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)

    Z = [Z1, Z2, Z3]
    A = [A1, A2, A3]

    cache = {"Z": Z, "A": A}
    return A3, cache


def loss(y_pred, y):
    m = y.shape[1]
    return -1/m*np.sum(np.sum(y*np.log(y_pred), axis=1), axis=0)


def back_propagate(X, Y, cache, params, mom_grads, learning_rate=0.01, beta=0.9):
    grads = calculate_gradients(X, Y, params, cache, mom_grads)
    W1, W2, W3 = params["W"]
    b1, b2, b3 = params["b"]
    dW1, dW2, dW3 = grads["dW"]
    db1, db2, db3 = grads["db"]
    W1 -= learning_rate*dW1
    W2 -= learning_rate*dW2
    W3 -= learning_rate*dW3
    b1 -= learning_rate*db1
    b2 -= learning_rate*db2
    b3 -= learning_rate*db3

    W = [W1, W2, W3]
    b = [b1, b2, b3]
    return {"W": W, "b": b}, grads


def accuracy(y_pred, y):
    m = y.shape[1]
    count = 0
    for i in range(m):
        pred = np.argmax(y_pred[:, i])
        res = np.argmax(y[:, i])
        if pred == res:
            count += 1
    return count/m


(x_train, y_train), (x_test, y_test) = mnist.load_data()

m, h, w = x_train.shape
m_test = x_test.shape[0]

# hyperparameters
epochs = 20
batch_size = 32
alpha_zero = 0.003
beta = 0.9
# 0 = no decay
decay_rate = 0

# inputs preprocessing
x_train = normalize_inputs(np.reshape(x_train, [m, h*w]).T)
y_train = one_hot(np.reshape(y_train, [1, m]))
x_test = normalize_inputs(np.reshape(x_test, [m_test, h*w]).T)
y_test = one_hot(np.reshape(y_test, [1, m_test]))
mom_grads = {"dW": [0, 0, 0], "db": [0, 0, 0]}

params = init_params([h*w, 128, 128, 10])
loss_train = []
ep = []

plt.plot([], [], [], [])
plt.ylabel('loss')
plt.xlabel('epochs')
plt.ion()
plt.show()

for i in range(epochs):
    alpha = alpha_zero/(1+decay_rate*i)
    randomize = np.arange(m)
    np.random.shuffle(randomize)
    X = x_train[:, randomize]
    Y = y_train[:, randomize]
    for j in range(m//batch_size):
        X_batch = X[:, batch_size * j:batch_size * (j + 1)]
        Y_batch = Y[:, batch_size * j:batch_size * (j + 1)]
        y_pred, cache = forward_prop(X_batch, params)
        params, mom_grads = back_propagate(X_batch, Y_batch, cache, params, mom_grads, learning_rate=alpha, beta=beta)
    y_pred, _ = forward_prop(x_train, params)
    y_test_pred, _ = forward_prop(x_test, params)
    l = loss(y_pred, y_train)
    acc = accuracy(y_pred, y_train)
    print("epoch: " + str(i) + " loss:" + str(l) + " accuracy:" + str(acc))
    loss_train.append(l)
    ep.append(i)
    plt.plot(ep, loss_train)
    plt.draw()
    plt.pause(0.001)

plt.savefig("train.png")
plt.close()
y_pred, _ = forward_prop(x_test, params)
acc = accuracy(y_pred, y_test)
print("accuracy on test set:", acc)
