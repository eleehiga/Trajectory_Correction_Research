# import tensorflow libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# construct LSTM neural network
def apply_nn(trajectory):
    start_predict = 0
    for i in range(len(trajectory)):
        if(trajectory[i] is None):
            start_predict = i
            break
    # scale data between 0 and 1
    cut_traj = trajectory[0:start_predict]
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_traj = scaler.fit_transform(np.array(cut_traj).reshape(-1,2))
    # make all X_train because just use that to predict points
    train_data = scaled_traj


    time_step=5
    X_train, y_train = create_dataset(train_data, time_step)

    model = Sequential()
    model.add(LSTM(time_step, input_shape=(x_train.shape[1:], 2), return_sequences=True)) # no activation as we are not returning a binary value
    model.add(LSTM(time_step, return_sequences=True))
    model.add(LSTM(time_step))


    model.add(Dense(2, activation='softmax')) # LSTM with return_sequence=False will return just one output so does Dense have to be 2

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(
        loss='mean_squared_error',
        optimizer=opt,
        metrics=['accuracy'],
    )
    model.summary()

    model.fit(x_train,
          y_train,
          epochs=100
          )

    #TODO make trajectory prediction
    
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
        if(t > del_start and del_end):
            trajectory.append([None, None])
        else:
            trajectory.append([x(t), y(t)])
    return trajectory

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

def main():
    print('NN Reconstruction')
    trajectory = load_test()
    corrected_traj = apply_nn(trajectory)
    plt.plot(corrected_traj)

if __name__ == '__main__':
    main()
