import numpy as np

cell_width = 1 # set value later
error_radius = 2 # set value later

def make_candidate_set(point):
    # find the maximum values of a and b such that they are less then the error radius  
    a_max = int(point[3] / (cell_width))
    size_set = int(np.pi * (error_radius/cell_width)^2) + 1 # figure out a better approximation later
    # use different values of a and b with like both being an integer between [-2,2]
    candidate_set = [[0 for i in range(3)] for j in range(size_set)]
    i = 0
    for j in list(-a_max, a_max+1):
        for k in list(-a_max, a_max+1):
            if((i**2 + j**2)**0.5 <= point[3]):
                if(i > candidate_set - 1):
                    break
                candidate_set[i][0] = j + point[0]
                candidate_set[i][1] = k + point[1]
                i = i + 1
    for k in len(candidate_set):
        candidate_set[k][2] = point[2]
    return candidate_set

def distance(first point, second_point):
    return ((first_point[0] - second_point[0])**2 + (first_point[1] - second_point[1])**2)**0.5
                    

def repair_distance_tendency(observation_point, candidate_point):
    # return the distance between these two points
    return distance(observation_point, candidate_point)


def travel_distance_tendency(prev_candidate, candidate):
    # return distance between the two points
    return distance(prev_candidate, candidate)

def speed_of_point(first_point, second_point):
    return distance(first_point, second_point)/(second_point[2] - second_point[2])

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
        denominator = denominator + np.exp(point, now)
    return numerator / (len(prev_candidate_set) * len(after candidate_set) * denominator)

def movement_score(prev_point, point, after_point, prev_candidate, candidate, after_candidate, prev_candidate_set, candidate_set, after_candidate_set):
    return normalize_repair(point, candidate, prev_candidate_set, candidate_set, after_candidate_set) + normalize_travel(prev_candidate, candidate, prev_candidate_set, candidate_set, after_candidate_set) + normalize_speed(prev_candidate, candidate, after_candidate, prev_candidate_set, candidate_set, after_candidate_set) 

def quality_repair(point, candidate, candidate_set):
    numerator = np.exp(distance(point, candidate))
    denominator = 0
    for now in candidate_set:
        denominator = denominator + np.exp(distane(point, now))
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
    for candidate in candidate_set
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
    # in trajector index - is x, 1 is y, 2 is time, and 3 is the error radius
    trajectory.append([0,0,len(trajectory)+1,error_radius]) 
    trajctory.insert(0,[0,0,-1,error_radius])
    candidate_set_list = []
    for point in trajectory:
        candidate_set_list.append(make_candidate_set(point))
    quality_set_list = []
    j = 1
    for candidate_set in candidate_set_list:
        quality_set_list.append(quality_candidates(candidate_set, trajectory[j-1], trajectory[j], trajectory[j+1]))
        j = j + 1

    F = [[] for j in range(len(trajectory)+2)] # make F a 3 dimensional 
    trace = [0]*(len(trajectory)+1)
    for i in range(2,len(trajectory)+1):
        for candidate in quality_set_list[i]:
            for prev_candidate in quality_set_list[i-1]:
                F = np.iinfo(im.dtype).max # machine limits for integer types, for floats do finfo
                for before_candidate in quality_set_list[i-2]:
                    l = movement_score(trajectory[i-2], trajectory[i-1], trajectory[i], before_candidate, prev_candidate, candidate, quality_set_list[i-2], quality_set_list[i-1], quality_set_list[i])
                    if F[i-1] + l < F[i]:
                        F[i] = F[i-1] + l
                        trace[i] = prev_candidate
    # choom p'n in Cn, p'n+1 in Cn+1 with minimum F(n+1,pn',p'n+1)
    
    return repaired_trajectory

def load_data(data)
    # put the constant radius on the trajectory data in index 3
    return trajectory

def main():
    trajectory = load_data('')
    dynamic_programing(trajectory, error_radius, cell_width)
    print("ATR Processing")

if __name__ == '__main__':
  main()
