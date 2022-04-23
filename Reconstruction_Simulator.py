import numpy as np
import os
import random
import copy
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import interpolate
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import pandas as pd

# 1. create trajectory with gaps and save the previous one also
# 2. train the neural network on this trajectory
# 3. run the forward only reconstruction
# 4. calculate the rms from the original
# 5. save the reconstructed graph, graph with gaps, and original graph
# 6. run the forward and backwards combined reconstruction
# 7. repeat steps 4 and 5
# 8. start from step 1 and keep doing this till reach specified amount of trajectories

num_traj= 1 # Dr. T wants 500 in total
offset = 0 # just in case want more runs, set this so ones before not overwritten
time_step = 10
gmin_x = 0
gmax_x = 0
gmin_y = 0
gmax_y = 0

points_length = 8
trajectory_length = 500
edge_protect = 0.05
max_coord_val = 500
num_gaps = 1
max_gaps = 5
max_gap_length = 20

def rand_point():
    return (max_coord_val*np.random.uniform(0,1), max_coord_val*np.random.uniform(0,1))

def rand_trajectory():
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
    no_gap_start = [] # no need to include end lengths here are already have edge protection 
    no_gap_stop = []
    for i in range(num_gaps):
        passes = False
        while(passes == False):
            passes = True # assume good in the beginning
            del_start = random.randint(edge_protect*trajectory_length,(1-edge_protect)*trajectory_length-max_gap_length) # -max_gap_length just in case
            if(len(no_gap_start) > 0):
                for k in range(len(no_gap_start)):
                    if(del_start > no_gap_start[k] and del_start < no_gap_stop[k]):
                        passes = False # if the del_start value is within the gaps
        del_stop = del_start+random.randint(max_gap_length/2,max_gap_length)
        for j in range(del_start, del_stop):
            out[0][j] = None
            out[1][j] = None
        no_gap_start.append(del_start-2*max_gap_length)
        no_gap_stop.append(del_stop+1*max_gap_length)
    trajectory = []
    for i in range(len(out[0])):
        trajectory.append([out[0][i], out[1][i]])
    perf_trajectory = []
    for i in range(len(out_prev[0])):
        perf_trajectory.append([out_prev[0][i], out_prev[1][i]])
    return trajectory, perf_trajectory

def train_nn(trajectory):
    min_x, max_x, min_y, max_y = min_max_from_nones(trajectory) # when I put this line outside the function, it will not be called
    print(min_x, max_x, min_y, max_y)
    global gmin_x
    gmin_x = min_x
    global gmin_y
    gmin_y = min_y
    global gmax_x
    gmax_x = max_x
    global gmax_y
    gmax_y = max_y
    print(gmin_x, gmax_x, gmin_y, gmax_y)
    start_predict = [] # start predicting there there are None in data
    end_predict = []
    for i in range(len(trajectory)):
        if((trajectory[i][0] is None or np.isnan(trajectory[i][0])) and len(start_predict) == len(end_predict)):
            start_predict.append(i) # if the trajectory is none and start and end array length the same
        elif(not(trajectory[i][0] is None or np.isnan(trajectory[i][0])) and len(start_predict) > len(end_predict)):
            end_predict.append(i) # will add to the end of the prediction array with it encountering non none points and more start than end for array length
    scaled_traj = [] # make an array with each bucket as a slice of the trajectory
    if(not(trajectory[0][0] is None)):
        cut_traj = trajectory[0:start_predict[0]]
        scaled_traj.append(scale_array(np.array(cut_traj).reshape(-1,2), min_x, max_x, min_y, max_y))
    for j in range(len(start_predict)-1):
        cut_traj = trajectory[end_predict[j]:start_predict[j+1]]
        scaled_traj.append(scale_array(np.array(cut_traj).reshape(-1,2), min_x, max_x, min_y, max_y))
    if(not(trajectory[len(trajectory)-1][0] is None)):
        cut_traj = trajectory[end_predict[len(end_predict)-1]:len(trajectory)]
        scaled_traj.append(scale_array(np.array(cut_traj).reshape(-1,2), min_x, max_x, min_y, max_y))
    # scale data between 0 and 1
    # make all X_train because just use that to predict points

    time_step=10 # how many points previously to use to predict
    X_train, y_train = create_dataset(scaled_traj, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    units = 3 # before 3 # for nodes in network
    model = Sequential()
    model.add(LSTM(units, input_shape=(time_step,2), return_sequences=True)) # no activation as we are not returning a binary value
    model.add(LSTM(units, return_sequences=True))
    model.add(LSTM(units))

    model.add(Dense(2)) # LSTM with return_sequence=False will return just one output so does Dense have to be 2

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(
        loss='mean_squared_error',
        optimizer=opt,
        metrics=['accuracy'],
    )
    #model.summary()

    model.fit(X_train,
          y_train,
          epochs=2
          ) # epochs=5000 is the best for forward only
            # 2500 is best for forward and backward
            # 500 is ok
    return model, scaled_traj, start_predict, end_predict # use this model and broken up scaled trajectory for evaluation

def forward_nn(trajectory, scaled_traj, start_predict, end_predict, model):
    print(gmin_x, gmax_x, gmin_y, gmax_y)
    min_x = gmin_x 
    min_y = gmin_y
    max_x = gmax_x
    max_y = gmax_y
    print(min_x, max_x, min_y, max_y)
    corrected_traj = copy.deepcopy(trajectory)
    # old way of fitting the curve
    for j in range(len(start_predict)):
        x_input = scaled_traj[j][scaled_traj[j].shape[0]-time_step:] # before first gap, last time_step points
        temp_input = list(x_input)
        i = 0
        while(i <= end_predict[j] - start_predict[j]):
            if(len(temp_input)>time_step): # will always be time_step + 1
                x_input=np.array(temp_input[1:]) # will remove one so its amount is still time_step
                x_input = x_input.reshape((1, time_step, 2))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat.tolist()) # append to the end
                temp_input=temp_input[1:]
                corrected_traj[start_predict[j] + i] = inv_scale_arr(yhat, min_x, max_x, min_y, max_y)[0]
                i=i+1
            else: # when len is time_step predict one more and add that on to the temp_input list
                x_input = x_input.reshape((1, time_step, 2))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat.tolist())
                corrected_traj[start_predict[j] + i] = inv_scale_arr(yhat, min_x, max_x, min_y, max_y)[0]
                i=i+1
    return corrected_traj # return forward corrected array

def for_back_nn(trajectory, scaled_traj, start_predict, end_predict, model):
    min_x = gmin_x 
    min_y = gmin_y
    max_x = gmax_x
    max_y = gmax_y
    corrected_traj = copy.deepcopy(trajectory)
    for j in range(len(start_predict)):
        x_input = scaled_traj[j][scaled_traj[j].shape[0]-time_step:] # before first gap, last time_step points
        temp_input = list(x_input)
        i = 0
        while(i <= end_predict[j] - start_predict[j]): # go till i <= because then can get scale 0 to use
            if(len(temp_input)>time_step): # will always be time_step + 1
                x_input=np.array(temp_input[1:]) # will remove one so its amount is still time_step
                x_input = x_input.reshape((1, time_step, 2))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat.tolist()) # append to the end
                temp_input=temp_input[1:]
                corrected_traj[start_predict[j] + i] = [yhat[0][0] * (1 - i/(end_predict[j] - start_predict[j])), yhat[0][1] * (1 - i/(end_predict[j] - start_predict[j]))] # do not inverse scale as this will be inverse scaled in backwards pass
                # do not do inverse scale here because will do that in backwards pass
                i=i+1
            else: # when len is time_step predict one more and add that on to the temp_input list
                x_input = x_input.reshape((1, time_step, 2))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat.tolist())
                corrected_traj[start_predict[j] + i] = [yhat[0][0] * (1 - i/(end_predict[j] - start_predict[j])), yhat[0][1] * (1 - i/(end_predict[j] - start_predict[j]))]
                i=i+1
    for j in range(len(start_predict)):
        # reverse the prediction
        # for every start and end predict value there are two scaled traj
        x_input = np.flipud(scaled_traj[j+1][0:time_step]) # after first gap, first time_step points, will reverse the python list
        temp_input = list(x_input)
        i = end_predict[j]
        while(i >= start_predict[j]):
            if(len(temp_input)>time_step): # will always be time_step + 1
                x_input=np.array(temp_input[1:]) # will remove one so its amount is still time_step
                x_input = x_input.reshape((1, time_step, 2))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat.tolist()) # append to the end
                temp_input=temp_input[1:]
                corrected_traj[i] = inv_scale_arr([[yhat[0][0] * (i - start_predict[j])/(end_predict[j] - start_predict[j]) + corrected_traj[i][0], yhat[0][1] * (i - start_predict[j])/(end_predict[j] - start_predict[j])+ corrected_traj[i][1]]], min_x, max_x, min_y, max_y)[0]
                i=i-1
            else: # when len is time_step predict one more and add that on to the temp_input list
                x_input = x_input.reshape((1, time_step, 2))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat.tolist())
                corrected_traj[i] = inv_scale_arr([[yhat[0][0] * (i - start_predict[j])/(end_predict[j] - start_predict[j]) + corrected_traj[i][0], yhat[0][1] * (i - start_predict[j])/(end_predict[j] - start_predict[j])+ corrected_traj[i][1]]], min_x, max_x, min_y, max_y)[0]
                i=i-1
    return corrected_traj

def min_max_from_nones(array):
    max_x = array[0][0]
    min_x = array[0][0]
    max_y = array[0][1]
    min_y = array[0][1]
    for i in range(len(array)):
        if(not(array[i][0] == None or np.isnan(array[i][0]))):
            if(array[i][0] > max_x):
                max_x = array[i][0]
            elif(array[i][0] < min_x):
                min_x = array[i][0]
            if(array[i][1] > max_y):
                max_y = array[i][1]
            elif(array[i][1] < min_y):
                min_y = array[i][1]
    return min_x, max_x, min_y, max_y

def scale_array(array, min_x, max_x, min_y, max_y):
    new_arr = []
    for i in range(len(array)):
        new_arr.extend([[0,0]])
        new_arr[i][0] = (array[i][0] - min_x)/(max_x - min_x)
        new_arr[i][1] = (array[i][1] - min_y)/(max_y - min_y)
    return np.array(new_arr)

def inv_scale_arr(array, min_x, max_x, min_y, max_y):
    new_arr = []
    for i in range(len(array)):
        new_arr.extend([[0,0]])
        new_arr[i][0]=array[i][0]*(max_x - min_x) + min_x
        new_arr[i][1]=array[i][1]*(max_y - min_y) + min_y
    return np.array(new_arr)

def get_rms(perf_traj, corrected_traj):
    rms = 0
    # loop will sum
    for i in range(len(perf_traj)):
        rms = np.sqrt((corrected_traj[i][0] - perf_traj[i][0])**2 + (corrected_traj[i][1] - perf_traj[i][1])**2) + rms
    return np.sqrt(rms / len(perf_traj))

def get_curvature_sum(trajectory):
    trajectory = np.array(trajectory) 
    # the [:,#] format cannot index python arrays, must be a np array 
    dx_dt = np.gradient(trajectory[:,0])
    dy_dt = np.gradient(trajectory[:,1])
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    return np.sum(curvature)

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)):
        for j in range(len(dataset[i])-time_step-1):
            a = dataset[i][j:(j+time_step)]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i][j + time_step])
        row = np.flipud(dataset[i])
        for j in range(len(dataset[i])-time_step-1):
            a = row[j:(j+time_step)]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(row[j + time_step])
    return np.array(dataX), np.array(dataY)

def extract_xy(trajectory):
    arr_x = []
    arr_y = []
    for t in range(len(trajectory)):
            arr_x.append(trajectory[t][0])
            arr_y.append(trajectory[t][1])
    return arr_x, arr_y

def main():
    print('Reconstruction Simulator')
    # uncomment if want to reset the csv file
    df = pd.DataFrame(columns=['forward rms','forward and backward rms', 'curvature sum', 'gaps amount'])
    df.to_csv('~/Documents/rms_curvature.csv', index=False)
    for j in range(1,max_gaps+1):
        for i in range(num_traj):
            num_gaps = j
            trajectory, perf_traj = rand_trajectory()
            model, scaled_traj, start_predict, end_predict = train_nn(trajectory)
            forward_traj = forward_nn(trajectory, scaled_traj, start_predict, end_predict, model)
            for_back_traj = for_back_nn(trajectory, scaled_traj, start_predict, end_predict, model)
            forward_x, forward_y = extract_xy(forward_traj)
            plt.scatter(forward_x, forward_y)
            plt.savefig(os.path.join(os.path.expanduser('~'), 'Documents/F_Images', 'f_im'+str(i+offset)+'.png'))
            for_back_x, for_back_y = extract_xy(for_back_traj)
            plt.scatter(for_back_x, for_back_y)
            plt.savefig(os.path.join(os.path.expanduser('~'), 'Documents/FB_Images', 'fb_im'+str(i+offset)+'.png'))
            plt.clf() # clear the entire figure
            df_tmp = pd.DataFrame([(get_rms(perf_traj, forward_traj), get_rms(perf_traj, for_back_traj), get_curvature_sum(perf_traj), num_gaps)])
            df_tmp.to_csv('~/Documents/rms_curvature.csv', mode='a', header=False, index=False)
            print("gaps, num traj")
            print(num_gaps, i)

if __name__ == '__main__':
    main()
