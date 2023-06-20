import numpy as np

inputV = [1.72, 1.23]
weights1 = [1.26, 0]
weights2 = [2.17, 0.32]

firstIndex = inputV[0] * weights1[0]
secondIndex = inputV[1] * weights1[1]
dotProduct = firstIndex + secondIndex
print(dotProduct)

dotProduct1 = np.dot(inputV, weights1)
print(dotProduct1)

