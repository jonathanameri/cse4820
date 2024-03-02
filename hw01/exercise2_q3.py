import numpy as np
import matplotlib.pyplot as plt

# Read the data from xvals.dat and create a matrix
xData = np.loadtxt('xvals.dat')
x = np.matrix(xData)

# Add a constant to the x matrix
x = np.insert(x, 0, 1, axis=1)

# Read the data from yvals.dat and create a matrix
yData = np.loadtxt('yvals.dat')
y = np.matrix(yData).T

# sigmoid function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# compute_gradient function
def compute_gradient(x, y, theta):
  z = np.dot(x, theta)
  gradient = np.dot(x.T, (y - sigmoid(z))) / y.size
  return gradient

# gradient ascent function
def gradient_ascent(x, y, learning_rate, tolerance):
  theta = np.zeros((x.shape[1], 1))
  gradient = compute_gradient(x, y, theta)
  numSteps = 0
  while np.linalg.norm(gradient) > tolerance:
    theta += learning_rate * gradient
    gradient = compute_gradient(x, y, theta)
    numSteps += 1
  return theta, numSteps

# run the gradient ascent algorithm
theta, numSteps = gradient_ascent(x, y, 0.01, 1e-6)

# print the result
print(theta)
print('numSteps:', numSteps)

# Assume data_x1 and data_x2 are the x1 and x2 values from your data
data_x1 = x[:,1]
data_x2 = x[:,2]

# Plot the data
plt.scatter(data_x1[y == 0], data_x2[y == 0], color='red')
plt.scatter(data_x1[y == 1], data_x2[y == 1], color='blue')

# Plot the decision boundary
x_values = np.linspace(min(data_x1), max(data_x1), num=100)
y_values = -(theta[0] + theta[1]*x_values) / theta[2]
plt.plot(x_values, y_values, color='green')

# Show the plot
plt.show()