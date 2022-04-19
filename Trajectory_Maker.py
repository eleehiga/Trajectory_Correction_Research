import numpy as np
import os
import random
import copy
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import interpolate

# plot a random path with random gaps in it
# 200 points
# x and y values 0 to 500 could be
# random gaps not in beginning or end but elsewhere
# gaps are about 20 long max
points_length = 8
trajectory_length = 500
edge_protect = 0.05
max_coord_val = 500
num_gaps = 7
max_gap_length = 20

def rand_point():
    return (max_coord_val*np.random.uniform(0,1), max_coord_val*np.random.uniform(0,1))

def main():
    points = []
    for i in range(points_length):
        points.append(rand_point())
        # random b-spline anchor points
    data = np.array(points)
    tck,u = interpolate.splprep(data.transpose(), s=0)
    unew = np.linspace(0,1,num=trajectory_length,endpoint=True)
    out = interpolate.splev(unew, tck)
    # place the gap after the edge_protection amount
    # set the end of the delete to be +rand from max_gap_length
    # delete the start and end of gap
    # make the next gaps not in between the previous gaps and has to be max gap_length away
    # make array of no_gap_start and no_gap_stop
    out_prev = copy.deepcopy(out)
    no_gap_start = []
    no_gap_stop = []
    for i in range(num_gaps):
        del_start = random.randint(edge_protect*trajectory_length,(1-edge_protect)*trajectory_length-max_gap_length)
        if(len(no_gap_start) > 0):
            for k in range(len(no_gap_start)):
                while(del_start > no_gap_start[k] and del_start < no_gap_stop[k]):
                    del_start = random.randint(edge_protect*trajectory_length,(1-edge_protect)*trajectory_length-max_gap_length)
        del_stop = del_start+random.randint(max_gap_length/2,max_gap_length)
        for j in range(del_start, del_stop):
            out[0][j] = None
            out[1][j] = None
        no_gap_start.append(del_start-2*max_gap_length)
        no_gap_stop.append(del_stop+max_gap_length)

    plt.figure()
    plt.plot(out[0], out[1], 'og')
    #plt.savefig(os.path.join(os.path.expanduser('~'), 'Documents/TCR_Images', 'fig1.png'))
    plt.show()
    #plt.plot(out_prev[0], out_prev[1], 'or')
    #plt.show()
    dx_dt = np.gradient(out_prev[0])
    dy_dt = np.gradient(out_prev[1])
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    print('curvature sum: ')
    print(np.sum(curvature))

if __name__=='__main__':
    main()
