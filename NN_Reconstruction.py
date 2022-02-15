# import tensorflow libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import array

# construct LSTM neural network
def apply_nn(trajectory):
    start_predict = [] # start predicting there there are None in data
    end_predict = []
    for i in range(len(trajectory)):
        if(trajectory[i][0] is None and len(start_predict) == len(end_predict)):
            start_predict.append(i) # if the trajectory is none and start and end array length the same
        elif(not(trajectory[i][0] is None) and len(start_predict) > len(end_predict)):
            end_predict.append(i) # will add to the end of the prediction array with it encountering non none points and more start than end for array length
    scaler=MinMaxScaler(feature_range=(0,1)) # will scale data between 0 and 1
    scaled_traj = [] # make an array with each bucket as a slice of the trajectory
    if(not(trajectory[0][0] is None)):
        cut_traj = trajectory[0:start_predict[0]]
        scaled_traj.append(scaler.fit_transform(np.array(cut_traj).reshape(-1,2)))
    for j in range(len(start_predict)-1):
        cut_traj = trajectory[end_predict[j]:start_predict[j+1]]
        scaled_traj.append(scaler.fit_transform(np.array(cut_traj).reshape(-1,2)))
    if(not(trajectory[len(trajectory)-1][0] is None)):
        cur_traj = trajectory[end_predict[len(end_predict)-1]:len(trajectory)]
        scaled_traj.append(scaler.fit_transform(np.array(cut_traj).reshape(-1,2)))
    # scale data between 0 and 1
    # make all X_train because just use that to predict points

    time_step=5 # how many points previously to use to predict
    X_train, y_train = create_dataset(scaled_traj, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    units = 3 # for nodes in network
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
          epochs=200
          ) # epochs=5000 is the best

    x_input = scaled_traj[0][scaled_traj[0].shape[0]-time_step:] # last time_step points
    temp_input = list(x_input)
    corrected_traj = trajectory # will replace null points
    for j in range(len(start_predict)):
        i = 0
        while(i < end_predict[j] - start_predict[j]):
            if(len(temp_input)>time_step): # will always be time_step + 1
                x_input=np.array(temp_input[1:]) # will remove one so its amount is still time_step
                x_input = x_input.reshape((1, time_step, 2))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat.tolist()) # append to the end
                temp_input=temp_input[1:]
                corrected_traj[start_predict[j] + i] = scaler.inverse_transform(yhat).tolist()[0] * (1 - i/(end_predict[j] - start_predict[j]))
                i=i+1
            else: # when len is time_step predict one more and add that on to the temp_input list
                x_input = x_input.reshape((1, time_step, 2))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat.tolist())
                corrected_traj[start_predict[j] + i] = scaler.inverse_transform(yhat).tolist()[0] * (1 - i/(end_predict[j] - start_predict[j]))
                i=i+1
    # reverse the trediction
    '''
    for j in range(len(start_predict)):
        i = end_predict[j] - start_predict[j]
        while(i >= 0):
            if(len(temp_input)>time_step): # will always be time_step + 1
                x_input=np.array(temp_input[1:]) # will remove one so its amount is still time_step
                x_input = x_input.reshape((1, time_step, 2))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat.tolist()) # append to the end
                temp_input=temp_input[1:]
                corrected_traj[start_predict[j] + i] = scaler.inverse_transform(yhat).tolist()[0] * (1 - i/(end_predict[j] - start_predict[j]))
                i=i-1
            else: # when len is time_step predict one more and add that on to the temp_input list
                x_input = x_input.reshape((1, time_step, 2))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat.tolist())
                corrected_traj[start_predict[j] + i] = scaler.inverse_transform(yhat).tolist()[0] * (1 - i/(end_predict[j] - start_predict[j]))
                i=i-1
    '''
    return corrected_traj


# reconstruction
    # take in points to train NN and then output for missing points

def load_test(): 
    trajectory_length = 100
    del_start = 80 # where to delete points
    del_end = 90
    # make a sine wave trajectory with holes, assuming data taken at even time variables
    # holes will have null with them
    x = lambda t : 0.0005*(t-1)*(t-100)*(t+100)
    y = lambda t : 2*t
    trajectory = []
    for t in range(trajectory_length):
        if(t > del_start and t < del_end):
            trajectory.append([None, None])
        else:
            trajectory.append([x(t), y(t)])
    return trajectory

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
    trajectory = load_test()
    corrected_traj = apply_nn(trajectory)
    arr_x,arr_y = extract_xy(corrected_traj)
    plt.plot(arr_x, arr_y)
    plt.show()

if __name__ == '__main__':
    main()
