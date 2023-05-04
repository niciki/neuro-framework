import numpy as np
np.random.seed(0)
import random

# Алгоритмы оптимизации

class Methods_optimizers:
    def optimizers(self, learning_rate, weights, bias, dweights, dbias):
        pass

# SGD (стохастический градиентный спуск)

class SGD(Methods_optimizers):
    def optimizers(self, learning_rate, weights, bias, dweights, dbias): 
        self.learning_rate = learning_rate
        weights -= self.learning_rate * dweights
        if bias is not None and dbias is not None:
            bias -= self.learning_rate * dbias
        return weights, bias

# Momentum SGD (градиентный спуск с моментом)

class MomentumSGD(Methods_optimizers):
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity_weights = 0
        self.velocity_bias = 0
    
    def optimizers(self, learning_rate, weights, bias, dweights, dbias):
        self.velocity_weights = 0
        self.velocity_bias = 0

        self.velocity_weights = (self.momentum * self.velocity_weights) - ((1 - self.momentum)* dweights)
        weights -= learning_rate * self.velocity_weights

        self.velocity_bias = (self.momentum * self.velocity_bias) - ((1 - self.momentum)* dbias)
        bias -= learning_rate * self.velocity_bias
        
        return weights, bias

# Gradient Clipping (ограничение градиента)

class GradientClipping(Methods_optimizers):
    def __init__(self, threshold):
        self.threshold = threshold
        
    def optimizers(self, learning_rate, weights, bias, dweights, dbias):
        clipped_gradients_weights = np.clip(dweights, -self.threshold, self.threshold)
        weights -= clipped_gradients_weights

        clipped_gradients_biases = np.clip(dbias, -self.threshold, self.threshold)
        bias -= clipped_gradients_biases
    
        return weights, bias

# Функции ошибки

# Для классификации


class CrossEntropyLoss:
    def forward(self,p,y):
        self.p = p
        self.y = y
        p_of_y = p[np.arange(len(y)), y]
        log_prob = np.log(p_of_y)
        return -log_prob.mean()

    def backward(self,loss):
        dlog_softmax = np.zeros_like(self.p)
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0/len(self.y)
        return dlog_softmax / self.p

class BinaryCrossEntropy:
  def forward(self, y, t):
    self.y = y
    self.t = t
    p_of_y = y[np.arange(len(t)), t]
    return (-(t * np.log(p_of_y) + (1 - t) * np.log(1 - p_of_y))).mean()

  def backward(self, loss):
    dbin = np.zeros_like(self.y)
    dbin[np.arange(len(self.t)), self.t] -= 1.0 / len(self.t)
    return (((1 - dbin) / (1 - self.y)) + (dbin / self.y))

# Для регрессии

class MeanSquaredError():
  def forward(self, y, t):
    self.y = y
    self.t = t
    return np.mean(np.power(self.y.flatten() - self.t, 2))

  def backward(self, loss):
    return 2 * (self.y - self.t)

class AbsoluteError:
    def forward(self, y, t):
        self.y = y
        self.t = t
        return np.mean(np.abs(self.y - self.t))
    
    def backward(self,loss):
        return np.where(self.y > self.t, 1, -1) / self.y.shape[0]

# Слои

class Net:
    def __init__(self,layers):
        self.layers = layers
            
    def forward(self,x):
        for l in self.layers:
            x = l.forward(x)
        return x
    
    def backward(self,z):
        for l in self.layers[::-1]:
            z = l.backward(z)
        return z
    
    def update(self,lr,optim):
        for l in self.layers:
            if 'update' in l.__dir__():
                l.update(lr,optim)

# Тут функции для обучения, в сам класс их включить не получилось

def get_loss_acc(self,x,y):
        p = self.net.forward(x)
        l = self.loss_func.forward(p,y)
        if self.task == "classification":
            pred = np.argmax(p,axis=1)
            acc = (pred==y).mean()
            return l,acc
        else:
            return l

def my_shuffle(train_x,train_y):
        x = train_x
        y = train_y
        a = np.random.permutation(len(train_x))
        for i in range(0, len(a), 1):
            train_x[i] = x[a[i]]
            train_y[i] = y[a[i]]

def train_epoch(self, train_x, train_y):

        for i in range(0,len(train_x),self.batch):
            xb = train_x[i:i+self.batch]
            yb = train_y[i:i+self.batch]

            p = self.net.forward(xb)
            l = self.loss_func.forward(p,yb)
            dp = self.loss_func.backward(l)
            dx = self.net.backward(dp)
            self.net.update(self.lr,self.optim)

        if self.task == "classification":
            loss, acc = get_loss_acc(self,train_x,train_y)
            print(f'Loss={loss}, accuracy={acc}: \n')
        else:
            loss = get_loss_acc(self,train_x,train_y)
            print(f'Loss={loss}: \n')

# Доступный функционал фреймворка

class NeuralNetwork:
    def create(self, layers, lossFunc, optim, epochsNumber, learning_rate, minibatch):
        self.net = Net(layers)
        self.loss_func = lossFunc
        self.optim = optim
        self.lr = learning_rate
        self.epochs_num = epochsNumber
        self.batch = minibatch
        if isinstance(lossFunc, MeanSquaredError | AbsoluteError):
            self.task = "regression"
        else:
            self.task = "classification"


    def fit(self, train_x, train_y):
        my_shuffle(train_x,train_y)
        for i in range(0, self.epochs_num, 1):
            print("Epoch ", i + 1, ":")
            train_epoch(self, train_x, train_y)


    def test(self,x,y):
        p = self.net.forward(x)
        l = self.loss_func.forward(p,y)
        if self.task == "classification":
            pred = np.argmax(p,axis=1)
            acc = (pred==y).mean()
            return pred, acc
        else:
            return l


    def predict(self,x):
        p = self.net.forward(x)
        if self.task == "classification":
            pred = np.argmax(p,axis=1)
            return pred
        else:
            return p

# Линейный слой

class Linear:
    def __init__(self,nin,nout):
        self.W = np.random.normal(0, 1.0/np.sqrt(nin), (nout, nin))
        self.b = np.zeros((1,nout))
        
    def forward(self, x):
        self.x = x
        return np.dot(x, self.W.T) + self.b
    
    def backward(self,dz):
        self.dW = np.dot(dz.T,self.x)
        self.db = dz.sum(axis=0)
        dx = np.dot(dz,self.W)
        return dx

    def update(self, lr, optim):
        self.W, self.b = optim.optimizers(lr, self.W, self.b, self.dW, self.db)

# Передаточные функции

class Softmax:
    def forward(self,z):
        self.z = z
        zmax = z.max(axis=1,keepdims=True)
        expz = np.exp(z-zmax)
        Z = expz.sum(axis=1,keepdims=True)
        return expz / Z

    def backward(self,dp):
        p = self.forward(self.z)
        pdp = p * dp
        return pdp - p * pdp.sum(axis=1, keepdims=True)

class Sigmoid:
    def forward(self, x):
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, dy):
        return dy * self.y * (1.0 - self.y)

class Threshold:
    def forward(self, x, threshold=0):
        self.y = np.zeros_like(x)
        self.y[x > threshold] = 1.0
        return self.y

    def backward(self, dy):
        return dy

class Tanh:
    def forward(self,x):
        y = np.tanh(x)
        self.y = y
        return y
    def backward(self,dy):
        return (1.0 - self.y ** 2) * dy
