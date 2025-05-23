import matplotlib.pyplot as plt
import numpy as np
from helper.opinion_dynamics import DGModel
import cvxpy as cp

def clicking_function(p, x):
    N = len(p)
    return 0.5*np.ones(N) + 0.5*p*x



def estimate_sensitivity(sim, simulation_steps, opinions, position):
    """
    Estimates the sensitvity with a Kalman filter
    :return: 
    """
    sensitivity_error = np.zeros(simulation_steps)

    sigma_r = 0.1
    sigma_q = 0.1

    H = np.eye(N)
    l = H.flatten(order='F')
    sigma = np.eye(N**2)
    K = np.zeros((N, N))
    R = sigma_r**2 * np.eye(N)
    Q = sigma_q**2 * np.eye(N**2)
    M = 10
    T = 20

    delta_x = np.zeros(N)
    delta_p = np.zeros(N)

    for i in range(1, simulation_steps + 1):
        opinions[i] = sim.update(p=position[i - 1])
        
        if(i % T == 0):
            CR_sensitivity_value = sim.get_CR_sensitivity(position[i - 1])
            print("CR_sensitivity_value \n", CR_sensitivity_value)

            delta_x = opinions[i] - opinions[i - T]

            kroeneker = np.kron(np.transpose(delta_p), np.eye(N))
            R = sigma_r**2 * np.eye(N)
            
            K = sigma @ np.transpose(kroeneker) @ np.linalg.inv(R + kroeneker @ sigma @ np.transpose(kroeneker))
            l = l + K @ (delta_x - kroeneker @ l)
            
            Q = sigma_q**2 * np.eye(N**2)

            sigma = sigma + Q - K @ kroeneker @ sigma

            H = np.reshape(l, (N, N), order='F')

            position[i] = np.random.uniform(-1, 1, N)

            delta_p = position[i] - position[i - 1]
        else:
            position[i] = position[i - 1]
    
    return sensitivity_error, sensitivity_error_diag, col_sum_error


# Parameters
simulation_steps = 500
N = 5
trials = 1
modes = np.array([2])

# Fixed matrix and parameters
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])

sensitivity_error = np.zeros((modes.size, trials))
sensitivity_error_diag = np.zeros((modes.size, trials))
col_sum_error = np.zeros((modes.size, trials))

s_err_temp = np.zeros((trials, simulation_steps))
s_err_diag_temp = np.zeros((trials, simulation_steps))
col_err_temp = np.zeros((trials, simulation_steps))

for t in range(trials):
    x_0 = np.random.uniform(-1, 1, N)
    gamma_p = np.random.uniform(0.01, 0.5, N)
    sim = DGModel(N=N, gamma=gamma_p, A=A, x_0=x_0)
        
    opinions = np.zeros((simulation_steps + 1, N))
    opinions[0] = x_0
    position = np.zeros((simulation_steps + 1, N))

    estimate_sensitivity(sim=sim, simulation_steps=simulation_steps, opinions=opinions, position=position)
    