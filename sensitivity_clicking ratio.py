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

    sigma_r = 0.1
    sigma_q = 0.005 #0.015

    S = np.eye(N)
    l = S.flatten(order='F')
    sigma = np.eye(N**2)
    K = np.zeros((N, N))
    R = sigma_r**2 * np.eye(N)
    Q = sigma_q**2 * np.eye(N**2)
    T = 60
    
    sensitivity_error = np.zeros(simulation_steps // T)

    least_squares_H = np.zeros(((simulation_steps // T)*N, N**2))
    least_squares_CR = np.zeros((simulation_steps // T)*N)

    last_CR = np.zeros(N)

    number_of_clicks = np.zeros(N)

    for i in range(1, simulation_steps + 1):
        opinions[i] = sim.update(p=position[i - 1])
        
        if(i % T == 0):
            delta_CR = number_of_clicks/40 - last_CR
            last_CR = number_of_clicks/40
            number_of_clicks = np.zeros(N)

            least_squares_CR[(i // T - 1) * N: i // T * N] = delta_CR

            #print("Position: ", position[i - 1])
            #print("CR_sensitivity_col_sum: ", np.diag(CR_sensitivity_value))
            #print("Postion: ", position[i - 1])
            H = build_delta_cr(position[i - 1], position[i - 1 - T])
            least_squares_H[(i // T - 1) * N: i // T * N, :] = H
            R = sigma_r**2 * np.eye(N)
            
            K = sigma @ np.transpose(H) @ np.linalg.inv(R + H @ sigma @ np.transpose(H))
            l = l + K @ (delta_CR - (H @ l))

            #print(delta_CR)
            #print(H @ l)
            
            Q = sigma_q**2 * np.eye(N**2)

            sigma = sigma + Q - K @ H @ sigma

            S = np.reshape(l, (N, N), order='F')

            #sensitivity_error[i-1] = np.mean(np.abs(np.diag(sim.get_sensitivity()) - np.diag(S)) / np.abs(np.diag(sim.get_sensitivity()))) * 100
            sensitivity_error[i // T - 1] = np.mean((sim.get_sensitivity() - S)**2)

            position[i] = np.random.uniform(-1, 1, N)
        
        elif(i % T >= 20):
            position[i] = position[i - 1]
            number_of_clicks += (np.random.rand(N) < clicking_function(position[i], opinions[i])).astype(int)
            
        else:
            position[i] = position[i - 1]
    
    print("original sensitivity: \n", sim.get_sensitivity(), "\n")

    S_least_squares = np.reshape(np.linalg.lstsq(least_squares_H, least_squares_CR)[0], (N, N), order='F')

    print(S)
    print(S_least_squares)

    ls_error = np.mean((sim.get_sensitivity() - S_least_squares)**2)
    #ls_error = np.mean(np.abs(np.diag(sim.get_sensitivity()) - np.diag(S_least_squares)) / np.abs(np.diag(sim.get_sensitivity()))) * 100

    plt.plot(sensitivity_error, linestyle='-', label='Kalman Filter Sensitivity Error')
    plt.plot(ls_error * np.ones(len(sensitivity_error)), linestyle='--', color='red', label='Least Squares Sensitivity Error')
    #plt.ylim(0,200)
    plt.grid()
    plt.legend()
    return S, S_least_squares


# Parameters
simulation_steps = 6000
N = 5
trials = 1
modes = np.array([2])

# Fixed matrix and parameters
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])

bias_kf = np.zeros(trials)
bias_ls = np.zeros(trials)
bias_kf_ls = np.zeros(trials)

for t in range(trials):
    x_0 = np.random.uniform(-1, 1, N)
    gamma_p = np.random.uniform(0.01, 0.5, N)
    sim = DGModel(N=N, gamma=gamma_p, A=A, x_0=x_0)
        
    opinions = np.zeros((simulation_steps + 1, N))
    opinions[0] = x_0
    position = np.zeros((simulation_steps + 1, N))

    S, S_LS = estimate_sensitivity(sim=sim, simulation_steps=simulation_steps, opinions=opinions, position=position)

    bias_kf[t] = np.mean(np.diag(S) - np.diag(sim.get_sensitivity()))
    bias_ls[t] = np.mean(np.diag(S_LS) - np.diag(sim.get_sensitivity()))

print(np.mean(bias_kf), bias_kf)
print(np.mean(bias_ls), bias_ls)

plt.show()