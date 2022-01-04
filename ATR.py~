import numpy as np
import matplotlib.pyplot as plt
import random

cell_width = 1 # set value later
error_radius = 2 # set value later
trajectory_length = 10

def make_candidate_set(point):
    # find the maximum values of a and b such that they are less then the error radius  
    a_max = int(point[3] / (cell_width))
    size_set = int(np.pi * (error_radius/cell_width)**2) + 1 # figure out a better approximation later
    # use different values of a and b with like both being an integer between [-2,2]
    candidate_set = [[0 for i in range(3)] for j in range(size_set)]
    i = 0
    for j in range(-a_max, a_max+1):
        for k in range(-a_max, a_max+1):
            if((k**2 + j**2)**0.5 <= point[3]):
                if(i > len(candidate_set) - 1):
                    i = 0 # reset i
                candidate_set[i][0] = j + point[0]
                candidate_set[i][1] = k + point[1]
                i = i + 1
    for k in range(len(candidate_set)):
        candidate_set[k][2] = point[2] # put the time in index 2 of the candidate set
    return candidate_set

def distance(first_point, second_point):
    return ((first_point[0] - second_point[0])**2 + (first_point[1] - second_point[1])**2)**0.5
                    

def repair_distance_tendency(observation_point, candidate_point):
    # return the distance between these two points
    return distance(observation_point, candidate_point)


def travel_distance_tendency(prev_candidate, candidate):
    # return distance between the two points
    return distance(prev_candidate, candidate)

def speed_of_point(first_point, second_point):
    if(second_point[2] - first_point[2] == 0):
        return np.iinfo(np.int32).max
    return distance(first_point, second_point)/(second_point[2] - first_point[2])

def speed_change_tendency(prev_candidate, candidate, after_candidate):
    # calculate the speech change of candidate
    return np.abs(speed_of_point(prev_candidate, candidate) - speed_of_point(candidate, after_candidate))

def normalize_speed(prev_candidate, candidate, after_candidate, prev_candidate_set, candidate_set, after_candidate_set):
    numerator = np.exp(speed_change_tendency(prev_candidate, candidate, after_candidate))
    denominator = 0
    for prev in prev_candidate_set:
        for now in candidate_set:
            for after in after_candidate_set:
                denominator = denominator + np.exp(speed_change_tendency(prev, now, after))
    return numerator / denominator

def normalize_travel(prev_candidate, candidate, prev_candidate_set, candidate_set, after_candidate_set):
    numerator = np.exp(travel_distance_tendency(prev_candidate, candidate))
    denominator = 0
    for prev in prev_candidate_set:
        for now in candidate_set:
            denominator = denominator + np.exp(travel_distance_tendency(prev, now))
    return numerator / (len(after_candidate_set) * denominator)

def normalize_repair(point, candidate, prev_candidate_set, candidate_set, after_candidate_set):
    numerator = np.exp(repair_distance_tendency(point, candidate))
    denominator = 0
    for now in candidate_set:
        denominator = denominator + np.exp(repair_distance_tendency(point, now))
    return numerator / (len(prev_candidate_set) * len(after_candidate_set) * denominator)

def movement_score(prev_point, point, after_point, prev_candidate, candidate, after_candidate, prev_candidate_set, candidate_set, after_candidate_set):
    return normalize_repair(point, candidate, prev_candidate_set, candidate_set, after_candidate_set) + normalize_travel(prev_candidate, candidate, prev_candidate_set, candidate_set, after_candidate_set) + normalize_speed(prev_candidate, candidate, after_candidate, prev_candidate_set, candidate_set, after_candidate_set) 

def quality_repair(point, candidate, candidate_set):
    numerator = np.exp(distance(point, candidate))
    denominator = 0
    for now in candidate_set:
        denominator = denominator + np.exp(distance(point, now))
    return numerator / denominator

def quality_travel(prev_point, candidate, candidate_set):
    numerator = np.exp(distance(prev_point, candidate))
    denominator = 0
    for now in candidate_set:
        denominator = denominator + np.exp(distance(prev_point, now))
    return numerator / denominator

def quality_speed(prev_point, candidate, after_point, candidate_set):
    numerator = np.exp(speed_change_tendency(prev_point, candidate, after_point))
    denominator = 0
    for now in candidate_set:
        denominator = denominator + np.exp(speed_change_tendency(prev_point, now, after_point))
    return numerator / denominator

def quality_candidates(candidate_set, prev_point, point, after_point):
    # get indices of candidates that pass quality repair, travel, and speed
    repair_candidates = []
    travel_candidates = []
    speed_candidates = []
    i = 0
    for candidate in candidate_set:
        if(quality_repair(point, candidate, candidate_set) <= quality_repair(point, point, candidate_set)):
            repair_candidates.append(i)
        if(quality_travel(prev_point, candidate, candidate_set) <= quality_travel(prev_point, point, candidate_set)):
            travel_candidates.append(i)
        if(quality_speed(prev_point, candidate, after_point, candidate_set) <= quality_speed(prev_point, point, after_point, candidate_set)):
            speed_candidates.append(i)
        i = i + 1
    # get the union between those sets
    union_candidates = list(set(repair_candidates) | set(travel_candidates) | set(speed_candidates))
    # return the candidates from this unioned set
    quality_set = []
    for j in union_candidates:
        quality_set.append(candidate_set[j])
    return quality_set

def dynamic_programming(trajectory, error_radius, cell_width):
    # in trajector index 0 is x, 1 is y, 2 is time, and 3 is the error radius
    trajectory.append([0,0,len(trajectory)+1,error_radius]) # maybe change the times later
    trajectory.insert(0,[0,0,-1,error_radius])
    candidate_set_list = []
    for point in trajectory:
        candidate_set_list.append(make_candidate_set(point))
    quality_set_list = []
    j = 0
    for candidate_set in candidate_set_list:
        if(j <= 0):
            quality_set_list.append(candidate_set);   
        elif(j > len(trajectory) - 2):
            quality_set_list.append(candidate_set);   
            break
        else:
            quality_set_list.append(quality_candidates(candidate_set, trajectory[j-1], trajectory[j], trajectory[j+1]))
        j = j + 1
    print(len(quality_set_list))
    #quality_set_list = candidate_set_list

    # below creates F(i-1, p'i-2, p'i) in index 0 and F(i, p'i-1, p'i) in index 1
    F = [[] for j in range(len(trajectory))] # make F be 2 trellises
    for i in range(1, len(trajectory)):
            for k in range(len(quality_set_list[i-1])):
                F[i].append([])
                for l in range(len(quality_set_list[i])):
                    F[i][k].append(0)
    '''
    # create F(1)
    F[1].append([])
    F[1].append([])
    for k in range(len(quality_set_list[0])):
        F[i][0].append([])
        for l in range(len(quality_set_list[i-1])):
            F[i][0][k].append(0)
        for k in range(len(quality_set_list[i-1])):
            F[i][1].append([])
            for l in range(len(quality_set_list[i])):
                F[i][1][k].append(0)

    # make the rest
    for i in range(2, len(trajectory)):
            F[i].append([])
            F[i].append([])
            for k in range(len(quality_set_list[i-2])):
                F[i][0].append([])
                for l in range(len(quality_set_list[i-1])):
                    F[i][0][k].append(0)
            for k in range(len(quality_set_list[i-1])):
                F[i][1].append([])
                for l in range(len(quality_set_list[i])):
                    F[i][1][k].append(0)
    # premake F
    '''

    # trace is trellis from p'i-1 to p'i. Values of traj and its index is stored
    trace = [[] for j in range(len(trajectory))] # trace should be a 3d array
    for i in range(1, len(trajectory)):
        for j in range(len(quality_set_list[i-1])):
            trace[i].append([])
            for k in range(len(quality_set_list[i])):
                trace[i][j].append(0)
    '''
    for i in range(2,len(trajectory)):
        j = 0
        for candidate in quality_set_list[i]:
            k = 0
            for prev_candidate in quality_set_list[i-1]:
                F[i][1][k][j] = np.iinfo(np.int32).max # machine limits for integer types, for floats do finfo
                n = 0
                for before_candidate in quality_set_list[i-2]:
                    l = movement_score(trajectory[i-2], trajectory[i-1], trajectory[i], before_candidate, prev_candidate, candidate, quality_set_list[i-2], quality_set_list[i-1], quality_set_list[i])
                    if F[i-1][0][n][k] + l < F[i][1][k][j]:
                        F[i][1][k][j] = F[i-1][0][n][k] + l
                        trace[i][k][j] = n
                    print(i,j,k,n)
                    n = n + 1
                k = k + 1
            j = j + 1
    # choose p'n in Cn, p'n+1 in Cn+1 with minimum F(n+1,pn',p'n+1)
    # Find the trajectory of F at at len(trajectory) + 1
    # At Fn is the culmination of all the Fs before
    min_pn = 0
    min_pn1 = 0
    min_Fn = np.iinfo(np.int32).max
    for j in range(0,len(F[len(trajectory)-1][1])):
        for k in range(0, len(F[len(trajectory)-1][1][j])):
            if(F[len(trajectory)-1][1][j][k] < min_Fn):
                min_pn = j
                min_pn1 = k
                min_Fn = F[len(trajectory)-1][1][j][k]
    '''
    for i in range(2,len(trajectory)):
        j = 0
        for candidate in quality_set_list[i]:
            k = 0
            for prev_candidate in quality_set_list[i-1]:
                F[i][k][j] = np.iinfo(np.int32).max # machine limits for integer types, for floats do finfo
                n = 0
                for before_candidate in quality_set_list[i-2]:
                    l = movement_score(trajectory[i-2], trajectory[i-1], trajectory[i], before_candidate, prev_candidate, candidate, quality_set_list[i-2], quality_set_list[i-1], quality_set_list[i])
                    if F[i-1][n][k] + l < F[i][k][j]:
                        F[i][k][j] = F[i-1][n][k] + l
                        trace[i][k][j] = n
                    print(i,j,k,n)
                    n = n + 1
                k = k + 1
            j = j + 1
    # choose p'n in Cn, p'n+1 in Cn+1 with minimum F(n+1,pn',p'n+1)
    # Find the trajectory of F at at len(trajectory) + 1
    # At Fn is the culmination of all the Fs before
    min_pn = 0
    min_pn1 = 0
    min_Fn = np.iinfo(np.int32).max
    for j in range(0,len(F[len(trajectory)-1])):
        for k in range(0, len(F[len(trajectory)-1][j])):
            if(F[len(trajectory)-1][j][k] < min_Fn):
                min_pn = j
                min_pn1 = k
                min_Fn = F[len(trajectory)-1][j][k]

    repaired_trajectory = [] # have to reverse it later as value will be put in reverse
    repaired_trajectory.append(quality_set_list[len(trajectory)-1][min_pn1])
    repaired_trajectory.append(quality_set_list[len(trajectory)-2][min_pn])
    repaired_trajectory.append(quality_set_list[len(trajectory)-3][trace[len(trajectory)-1][min_pn][min_pn1]])
    
    trace_pi1 = trace[len(trajectory)-1][min_pn][min_pn1] #finds n-2, this here is i -1
    trace_pi = min_pn
    for i in range(len(trajectory)-4, 0, -1):
        pi2 = trace[i][trace_pi1][trace_pi] # get i -2
        print(i,pi2)
        repaired_trajectory.append(quality_set_list[i][pi2])
        # shift the shift pis down one
        trace_pi = trace_pi1
        trace_pi1 = pi2
    repaired_trajectory.append(quality_set_list[0][0])
    repaired_trajectory.reverse()
    
    return repaired_trajectory

def load_data(data):
    return 0

def load_test(): # make sine wave
    # put the constant radius on the trajectory data in index 3
    # trajectory = [[j,j,j,1] for j in range(40)] # make F be 2 trellises
        # above makes a straight line trajectory
    x = lambda t : 0.0005*(t-1)*(t-100)*(t+100)
    y = lambda t : 2*t
    arr_x = []
    arr_y = []
    trajectory = []
    for t in range(trajectory_length):
        arr_x.append(x(t))
        arr_y.append(y(t))
        trajectory.append([x(t), y(t), t, 1])
    #for t in range(10, 30, 5):
    #    arr_x[t] = random.randrange(-3,3) + arr_x[t]
    #    arr_y[t] = random.randrange(-3,3) + arr_y[t]
    #    trajectory[t][0] = arr_x[t]
    #    trajectory[t][1] = arr_y[t]
    arr_x[4] = random.randrange(-1,1) + arr_x[4]
    arr_y[4] = random.randrange(-1,1) + arr_y[4]
    trajectory[4][0] = arr_x[4]
    trajectory[4][1] = arr_y[4]
    return trajectory, arr_x, arr_y

def extract_xy(trajectory):
    arr_x = []
    arr_y = []
    for t in range(len(trajectory)):
        arr_x.append(trajectory[t][0])
        arr_y.append(trajectory[t][1])
    return arr_x, arr_y

def main():
    trajectory, arr_x, arr_y = load_test()
    repaired_trajectory = dynamic_programming(trajectory, error_radius, cell_width)
    before_x, before_y = extract_xy(trajectory)
    after_x, after_y = extract_xy(repaired_trajectory)

    plt.figure(1)
    plt.scatter(after_x, after_y, label = "line 2")
    plt.figure(2)
    plt.scatter(before_x, before_y, label = "line 1")
    plt.legend()
    plt.show()
    print(trajectory)
    print("ATR Processing")
    print(repaired_trajectory)

if __name__ == '__main__':
  main()
