# Trajectory_Correction_Research
Research Project under the mentorship of Dr. Tadrous. The goal of this project is to create an offline algorithm that can take in a trajectory with corrupted (location data off from what it actually should be) or missing (but still having the time stamp) data points and to correct a trajectory and provide metrics about the correction.

ATR.py: Is a python version of the Automatic Trajectory Repair algorithm (https://www.researchgate.net/publication/338165989_ATR_Automatic_Trajectory_Repairing_With_Movement_Tendencies) in which this project will modify this algorithm to account for missing data points

NN_Reconstruction.py: Is a python implementation of an LSTM Neural Network to predict trajectory coordinates from already known data points (coded from ideas in this research paper https://ieeexplore.ieee.org/document/8713215).

Traj_Research_Report.pdf: Is the report for this project describe the methodology and results. 
