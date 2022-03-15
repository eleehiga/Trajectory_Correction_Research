import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#x1 = [100, 88,  67,  50,  35,  27, 18,  11,  8,  5,  4,  2]
#y1 = [0., 13.99, 27.99, 41.98, 55.98, 69.97, 83.97, 97.97, 111.96, 125.96, 139.95, 153.95]
x1 = [10, 100, 200, 310, 400, 500, 300, 0]
y1 = [90, 300, 100, 0, 110, 200, 400, 500]

# Combine lists into list of tuples
points = zip(x1, y1)

# Sort list of tuples by x-value
#points = sorted(points, key=lambda point: point[0])

# Split list of tuples into two list of x values any y values
x1, y1 = zip(*points)

new_length = 500
new_x = np.linspace(min(x1), max(x1), new_length)
new_y = sp.interpolate.interp1d(x1, y1, kind='cubic')(new_x)

plt.scatter(new_x, new_y)
plt.show()
