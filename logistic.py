import numpy as np
import matplotlib.pyplot as plt

def sigmoid(score):
    return 1 / (1 + np.exp(-score))

def calculate_error(line_parameters, points, y):
    m = points.shape[0]
    linear_combination = points @ line_parameters
    p = sigmoid(linear_combination)
    cross_entropy = -(1 / m) * np.sum(np.multiply(y, np.log(p)) + np.multiply(1 - y, np.log(1 - p)))
    return cross_entropy

def draw(ax, x1, x2, line=None):
    if line is None:
        line, = ax.plot(x1, x2, 'k-')  # Create a new line
    else:
        line.set_xdata(x1)
        line.set_ydata(x2)
    plt.pause(0.01)
    return line

def gradient_decent(line_parameters, points, y, alpha):
    m = points.shape[0]
    line = None
    for i in range(2500):
        linear_combination = points @ line_parameters
        p = sigmoid(linear_combination)
        gradient = (points.T @ (p - y)) * (alpha / m)
        line_parameters -= gradient

        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)

        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -(b + w1 * x1) / w2

        line = draw(ax, x1, x2, line)
        print(f"Iteration {i+1}: Error = {calculate_error(line_parameters, points, y)}")

n_pts = 500
np.random.seed(0)
bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))
line_parameters = np.zeros((3, 1))

y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts * 2, 1)

_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')

gradient_decent(line_parameters, all_points, y, 0.06)
plt.show()
