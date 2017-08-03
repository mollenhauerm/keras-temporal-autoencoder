import numpy as np
from scipy.stats import rv_discrete

def noise(state, cov = [[30, 0], [0, .02]]):
    # emission probabilities of HMM
    return np.random.multivariate_normal(state,cov= cov)
    

def generate_trajectory(T,states=[np.array([0,3]),np.array([0,5])],trans_prob = [.95,.05],start_state=0):
    # yields a T-timestep realization of 2-state HMM
    # traj will contain 2d-trajectory data, clusters will contain 
	# labeling of our metastable states
    traj = np.array(states[start_state])
    clusters=[]
    
    state0prob = rv_discrete(values=([0,1],trans_prob))
    state1prob = rv_discrete(values=([0,1],trans_prob[::-1]))
    active_state = start_state
    clusters.append(start_state)
    for i in range(T):
        if active_state == 0:
            active_state = state0prob.rvs(size=1)[0]
            traj = np.vstack([traj,noise(states[active_state])])
            clusters.append(active_state)
        if active_state == 1:
            active_state = state1prob.rvs(size=1)[0]
            traj = np.vstack([traj,noise(states[active_state])])
            clusters.append(active_state)
    return traj,np.array(clusters)
