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

# Value of theta from previous part
theta = np.array([[-2.62045303], [0.76035866], [1.17194243]])

# Convert x and y to numpy arrays
x = np.array(x)
y = np.array(y)

# Plot the data
for label in [0, 1]:
  # Select rows where the label matches, and plot the corresponding x1 and x2
  plt.scatter(x[y.flatten() == label, 1], x[y.flatten() == label, 2], label=f'Label {label}')

# Plotting the decision boundary
# Using the equation theta0 + theta1 * x1 + theta2 * x2 = 0, we can solve for x2
# x2 = -(theta0 + theta1 * x1) / theta2
# Determine the range of x1 values
x1_values = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
# Calculate x2 values using the decision boundary equation
x2_values = -(theta[0] + theta[1] * x1_values) / theta[2]

plt.plot(x1_values, x2_values, label='Decision Boundary', color='red')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Data with Decision Boundary')
plt.show()