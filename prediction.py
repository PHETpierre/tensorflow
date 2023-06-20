import numpy as np
inputV = np.array([2, 1.5])
weights1 = np.array([1.45, -0.66])
bias = np.array([0.0])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def makePrediction(inputV, weights, bias):
    layer1 = np.dot(inputV, weights) + bias
    layer2 = sigmoid(layer1)
    return layer2

prediction = makePrediction(inputV, weights1, bias)
print(prediction)

target = 0
mse = np.square(prediction - target)
print(mse)

deriv = 2 * (prediction - target)
print(deriv)

weights1 = weights1 - deriv
prediction = makePrediction(inputV, weights1, bias)
error = (prediction - target) ** 2

print("prediction:", prediction ,"error:", error)

def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))

derror_dprediction = 2 * (prediction - target)
layer1 = np.dot(inputV, weights1) + bias
dprediction_dlayer1 = sigmoid_deriv(layer1)
dlayer1_dbias = 1

derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
