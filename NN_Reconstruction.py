# import tensorflow libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import array
import scipy as sp
from scipy.interpolate import interp1d

# construct LSTM neural network
def apply_nn(trajectory):
    min_x, max_x, min_y, max_y = min_max_from_nones(trajectory)
    start_predict = [] # start predicting there there are None in data
    end_predict = []
    for i in range(len(trajectory)):
        if(trajectory[i][0] is None and len(start_predict) == len(end_predict)):
            start_predict.append(i) # if the trajectory is none and start and end array length the same
        elif(not(trajectory[i][0] is None) and len(start_predict) > len(end_predict)):
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
    model.summary()

    model.fit(X_train,
          y_train,
          epochs=5000
          ) # epochs=5000 is the best for forward only
            # 2500 is best for forward and backward
            # 500 is ok
    
    corrected_traj = trajectory # will replace null points
    ''' 
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
    '''
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
    
    '''# old way but reversed
    for j in range(len(start_predict)):
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
                corrected_traj[i] = inv_scale_arr(yhat, min_x, max_x, min_y, max_y)[0]
                i=i-1
            else: # when len is time_step predict one more and add that on to the temp_input list
                x_input = x_input.reshape((1, time_step, 2))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat.tolist())
                corrected_traj[i] = inv_scale_arr(yhat, min_x, max_x, min_y, max_y)[0]
                i=i-1
    '''
    return corrected_traj
# reconstruction
    # take in points to train NN and then output for missing points

def min_max_from_nones(array):
    max_x = array[0][0]
    min_x = array[0][0]
    max_y = array[0][1]
    min_y = array[0][1]
    for i in range(len(array)):
        if(array[i][0] != None):
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

def load_test(): 
    del_start = [40,80,140, 200, 300, 400, 450] # where to delete points
    del_end = [50,90,160, 220,350, 410, 460]
    '''
    trajectory_length = 100
    # make a sine wave trajectory with holes, assuming data taken at even time variables
    # holes will have null with them
    x = lambda t : 0.0005*(t-1)*(t-100)*(t+100)
    y = lambda t : 2*t
    trajectory = []
    '''
    # making spline
    x1 = [10, 100, 200, 310, 400, 500, 300, 0]
    y1 = [90, 300, 100, 0, 110, 200, 400, 500]

    new_length = 500
    new_x = np.linspace(min(x1), max(x1), new_length)
    new_y = sp.interpolate.interp1d(x1, y1, kind='cubic')(new_x)
    points = list(zip(new_x, new_y))
    trajectory = [list(item) for item in points]
    perf_traj = trajectory.copy()
    for t in range(len(del_start)):
        for i in range(del_start[t], del_end[t]):
            trajectory[i] = [None,None]

    ''' # for the 100 length trajectory
    for t in range(trajectory_length):
        if(t > del_start and t < del_end):
            trajectory.append([None, None])
        else:
            trajectory.append([x(t), y(t)])
    '''
    #print(trajectory)
    return trajectory, perf_traj

def get_rms(perf_traj, corrected_traj):
    rms = 0
    # loop will sum
    for i in range(len(perf_traj)):
        rms = np.sqrt((corrected_traj[i][0] - perf_traj[i][0])**2 + (corrected_traj[i][1] - perf_traj[i][1])**2) + rms
    return np.sqrt(rms / len(perf_traj))

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
    print('NN Reconstruction')
    trajectory, perf_traj = load_test()
    #arr_x,arr_y = extract_xy(trajectory)
    #plt.scatter(arr_x, arr_y)
    #plt.show()
    #exit() 
    corrected_traj = apply_nn(trajectory)
    arr_x,arr_y = extract_xy(corrected_traj)
    print('rms:')
    print(get_rms(perf_traj, corrected_traj))
    plt.scatter(arr_x, arr_y)
    plt.show()

if __name__ == '__main__':
    main()
