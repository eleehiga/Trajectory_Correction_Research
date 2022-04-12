# 1. create trajectory with gaps and save the previous one also
# 2. train the neural network on this trajectory
# 3. run the forward only reconstruction
# 4. calculate the rms from the original
# 5. save the reconstructed graph, graph with gaps, and original graph
# 6. run the forward and backwards combined reconstruction
# 7. repeat steps 4 and 5
# 8. start from step 1 and keep doing this till reach specified amount of trajectories

num_trajectories = 1 # Dr. T wants 500
time_step = 10
min_x, max_x, min_y, max_y = 0

points_length = 8
trajectory_length = 500
edge_protect = 0.05
max_coord_val = 500
num_gaps = 7
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
    trajectory = []
    for i in range(len(out)):
        trajectory.append([out[0][i], out[1][i]])
    perf_trajectory = []
    for i in range(len(out_prev)):
        trajectory.append([out_prev[0][i], out_prev[1][i]])
    return trajectory, perf_trajectory

def train_nn(trajectory):
    return 0

def forward_nn(trajectory, scaled_traj, model):
    return 0

def for_back_nn(trajectory, scaled_traj, model):
    return 0

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
    print('Reconstruction Simulator')
    trajectory, perf_trajectory = rand_trajectory()
    min_x, max_x, min_y, max_y = min_max_from_nones(trajectory)
    model, scaled_traj = train_nn(trajectory)
    forward_traj = forward_nn(trajectory, scaled_traj, model)
    for_back_traj = for_back_nn(trajectory, scaled_traj, model)
    forward_x, forward_y = extract_xy(forward_traj)
    print('forward rms:')
    print(get_rms(perf_traj, forward_traj))
    plt.scatter(forward_x, forward_y)
    plt.show()
    for_back_x, for_back_y = extract_xy(for_back_traj)
    print('forward and backward rms:')
    print(get_rms(perf_traj, for_back_traj))
    plt.scatter(for_back_x, for_back_y)
    plt.show()

if __name__ == '__main__':
    main()
