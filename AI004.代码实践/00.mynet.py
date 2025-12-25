import math
import random

class Value:
    def __init__(self,data,_children=(),op="",lable=""):
        self.data = data
        self._prev = set(_children)
        self._op = op
        self.lable = lable
        self.grad = 0.0
        self._backward = lambda : None
    
    def __repr__(self):
        return f"Value(data={self.data:0.4f})"
    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data,(self,other),"+")
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out
    
    def __radd__(self,other):
        return self + other
    
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data,(self,other),"*")
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self,other):
        return self * other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) -1) / (math.exp(2*x) +1)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s,(self,),'sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self,),'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int,float))
        out = Value(self.data**other,(self,),f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1
    
    def __rneg__(self):
        return self * -1
    
    def __rsub__(self,other):
        return other + (-self)

    def __sub__(self,other):
        return self + (-other)

    def backward(self):
        
        topo = []
        visited = set()
        def search(n):
            if n not in visited:
                visited.add(n)
                for child in n._prev:
                    search(child)
                topo.append(n)
        search(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1), lable=f"w{i}") for i in range(nin)]
        self.b = Value(0, lable="b")
    
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out if len(out) >1 else out[0]
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
# 0.定义训练数据
xs = [[0,0,0,0,0,0,0,0],
      [1,0,0,0,0,0,0,0],
      [0,1,0,0,0,0,0,0],
      [0,0,1,0,0,0,0,0],
      [1,1,0,0,0,0,0,0],
      [1,0,1,0,0,0,0,0],
      [0,1,1,0,0,0,0,0],
      [1,1,1,0,0,0,0,0],
      [0,0,0,1,0,0,0,0],
      [1,0,0,1,0,1,0,0],
      [0,0,0,0,1,0,0,0],
      [0,0,1,0,1,0,0,1],
      [0,0,0,0,0,1,0,0],
      [0,0,0,0,0,0,1,0],
      [0,0,0,0,0,0,0,1],
      [0,0,0,0,0,1,1,0],
      [0,0,0,0,0,1,0,1],
      [0,0,0,0,0,1,1,1],
      [1,1,1,1,1,1,1,1]]

ys = [[1,0,0,0],
      [1,0,0,0],
      [1,0,0,0],
      [1,0,0,0],
      [1,0,0,0],
      [1,0,0,0],
      [1,0,0,0],
      [1,0,0,0],
      [0,1,0,0],
      [0,1,0,0],
      [0,0,0,1],
      [0,0,0,1],
      [0,0,1,0],
      [0,0,1,0],
      [0,0,1,0],
      [0,0,1,0],
      [0,0,1,0],
      [0,0,1,0],
      [0,0,1,0]]

# 1.定义神经网络结构
mlp = MLP(8,[5,4])

# 2.训练神经网络
for epoch in range(1000):
    # 1.前向计算
    y_preds = []
    for x in xs:
        y_pred = mlp(x)
        y_preds.append(y_pred)
    # 2.计算均方误差损失   
    loss = sum((sum((yt - yp)**2 for yt, yp in zip(y_target, y_pred)) for y_pred, y_target in zip(y_preds, ys)))
    print(f"Epoch {epoch}. loss: {loss.data}")
    # 3.初始化梯度
    for p in mlp.parameters():
        p.grad = 0.0
    # 4.反向传播
    loss.backward()
    # 5.调整参数
    learning_rate = 0.02
    for p in mlp.parameters():
        p.data += -learning_rate * p.grad

print("Final predictions:")
for x in xs:    
    y_pred = mlp(x)
    print(y_pred)
