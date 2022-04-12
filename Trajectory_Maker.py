import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import interpolate

# 1d interpolation
'''
#x1 = [100, 88,  67,  50,  35,  27, 18,  11,  8,  5,  4,  2]
#y1 = [0., 13.99, 27.99, 41.98, 55.98, 69.97, 83.97, 97.97, 111.96, 125.96, 139.95, 153.95]
#x1 = [10, 100, 200, 310, 400, 500, 300, 0]
#y1 = [90, 300, 100, 0, 110, 200, 400, 500]
x1 = [100, 200,10, 310, 400, 500, 300, 0]
y1 = [300, 100, 90, 0, 110, 200, 400, 500]

# Combine lists into list of tuples
#points = zip(x1, y1)

# Sort list of tuples by x-value
#points = sorted(points, key=lambda point: point[0])

# Split list of tuples into two list of x values any y values
#x1, y1 = zip(*points)

new_length = 500
new_x = np.linspace(min(x1), max(x1), new_length)
new_y = sp.interpolate.interp1d(x1, y1, kind='cubic')(new_x)

plt.scatter(new_x, new_y)
plt.show()
'''
#splprep
points = [(100,300),(200,100),(10,90),(310,0),(400,110),(500,200),(300,400),(0,500)]
print(points)
data = np.array(points)

tck,u = interpolate.splprep(data.transpose(), s=0)
print('tck: ')
print(tck)
#unew = np.arange(0, 1.01, 0.01)
unew = np.linspace(0,1,num=200,endpoint=True)
print('unew: ')
print(unew)
#leads to redicuously big numbers -> unew = np.array(range(0,500))
out = interpolate.splev(unew, tck)
print('out: ')
print(out) # out is a 2d array with the first array for x values and second for y values
print('len(out[0]): ')
print(len(out[0]))

plt.figure()
plt.plot(out[0], out[1], color='orange')
plt.plot(data[:,0], data[:,1], 'ob')
plt.show()
