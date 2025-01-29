from models.modelsFDH import FDH
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from mpl_toolkits.mplot3d import Axes3D

X = np.array([[1, 4], [2, 2], [3, 1], [2, 6], [3, 4], [4, 3], [5, 1], [4, 8], [5, 5], [6, 4], [7, 2]])
y = np.array([2, 1, 3, 5, 1, 2, 1, 10, 6, 4, 7])
y = y.reshape((11, 1))
fdh = FDH(X, y)
fdh.fdh_output_vrs()
x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
x2_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)

x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T

y_grid = np.array([fdh.predict([x1, x2], 'fdh_output_vrs') for x1, x2 in grid_points])

y_grid = np.ma.masked_where(y_grid == 0, y_grid)

y_grid = y_grid.reshape(x1_grid.shape)

fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x1_grid, x2_grid, y_grid, cmap='Spectral', alpha=1)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title("3D Plot with CART Decision Frontier")
ax.legend()

plt.show()