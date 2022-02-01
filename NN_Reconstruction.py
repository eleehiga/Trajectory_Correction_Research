# import tensorflow libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# construct LSTM neural network
def make_nn(traj):
    #TODO make x_train, y_train, x_test, y_test
    # x_test, y_test is last 5%, x_train, y_train is before that
    
    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )

    model.fit(x_train,
          y_train,
          epochs=3,
          validation_data=(x_test, y_test))


# reconstruction
    # take in points to train NN and then output for missing points

def load_test(): 
    # make a sine wave trajectory with holes, assuming data taken at even time variables
    # holes will have null with them
    x = lambda t : 0.0005*(t-1)*(t-100)*(t+100)
    y = lambda t : 2*t
    arr_x = []
    arr_y = []
    trajectory = []
    for t in range(trajectory_length):
        arr_x.append(x(t))
        arr_y.append(y(t))
        trajectory.append([x(t), y(t), t, 1])
    return trajectory, arr_x, arr_y

def main():
    print('test')

if __name__ == '__main__':
    main()
