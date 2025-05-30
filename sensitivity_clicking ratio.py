import matplotlib.pyplot as plt
import numpy as np
from helper.opinion_dynamics import DGModel
from helper.build_delta_cr import build_delta_cr
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
    sigma_q = 0.015

    H = np.eye(N)
    l = H.flatten(order='F')
    sigma = np.eye(N**2)
    K = np.zeros((N, N))
    R = sigma_r**2 * np.eye(N)
    Q = sigma_q**2 * np.eye(N**2)
    M = 10
    T = 60

    delta_p = np.zeros(N)
    last_CR = np.zeros(N)

    number_of_clicks = np.zeros(N)

    for i in range(1, simulation_steps + 1):
        opinions[i] = sim.update(p=position[i - 1])
        
        if(i % T == 0):
            delta_CR = number_of_clicks/40 - last_CR
            last_CR = number_of_clicks/40
            number_of_clicks = np.zeros(N)

            CR_sensitivity_value = sim.get_CR_sensitivity(position[i - 1])
            #print("Position: ", position[i - 1])
            #print("CR_sensitivity_col_sum: ", np.diag(CR_sensitivity_value))
            #print("Postion: ", position[i - 1])
            H = build_delta_cr(position[i - 1], position[i - 1 - T])
            R = sigma_r**2 * np.eye(N)
            
            K = sigma @ np.transpose(H) @ np.linalg.inv(R + H @ sigma @ np.transpose(H))
            l = l + K @ (delta_CR - (H @ l))

            print(delta_CR)
            print(H @ l)
            
            Q = sigma_q**2 * np.eye(N**2)

            sigma = sigma + Q - K @ H @ sigma

            H = np.reshape(l, (N, N), order='F')

            print("estimation: ", np.diag(H))
            print("original sensitivity: ", np.diag(sim.get_sensitivity()), "\n")
            sensitivity_error[i-1] = np.mean(np.abs(np.diag(sim.get_sensitivity()) - np.diag(H)) / np.abs(np.diag(sim.get_sensitivity()))) * 100

            position[i] = np.random.uniform(-1, 1, N)

            delta_p = position[i] - position[i - 1]
        
        elif(i % T >= 20):
            position[i] = position[i - 1]
            number_of_clicks += (np.random.rand(N) < clicking_function(position[i], opinions[i])).astype(int)
            
        else:
            position[i] = position[i - 1]
    
    plt.plot(sensitivity_error)
    plt.ylim(0,200)
    
    return sensitivity_error, sensitivity_error_diag, col_sum_error


# Parameters
simulation_steps = 6000
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

plt.show()