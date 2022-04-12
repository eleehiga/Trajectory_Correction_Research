# 1. create trajectory with gaps and save the previous one also
# 2. train the neural network on this trajectory
# 3. run the forward only reconstruction
# 4. calculate the rms from the original
# 5. save the reconstructed graph, graph with gaps, and original graph
# 6. run the forward and backwards combined reconstruction
# 7. repeat steps 4 and 5
# 8. start from step 1 and keep doing this till reach specified amount of trajectories
