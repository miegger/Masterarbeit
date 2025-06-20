import matplotlib.pyplot as plt
import numpy as np
from helper.opinion_dynamics import DGModel
from helper.build_delta_cr import build_delta_cr
from scipy.special import expit  # Sigmoid
from scipy.optimize import minimize_scalar


def clicking_function(p, x):
    N = len(p)
    return 0.5*np.ones(N) + 0.5*p*x


def neg_log_likelihood(alpha, opinions, positions, clicks):
    total_loss = 0.0
    for i, o in enumerate(opinions):
        for j, p in enumerate(positions):
            d = abs(o - p)
            prob = expit(-alpha * d)
            c = clicks[i, j]
            # Add log-likelihood contribution (with small eps for numerical stability)
            prob = np.clip(prob, 1e-6, 1 - 1e-6)
            total_loss += c * np.log(prob) + (1 - c) * np.log(1 - prob)
    return -total_loss  # Minimize negative log-likelihood


def estimate_sensitivity(sim, simulation_steps, opinions, position):
    """
    Estimates the sensitvity with a Kalman filter
    :return: 
    """
    sensitivity_error = np.zeros(simulation_steps)

    sigma_r = 0.1
    sigma_q = 0.005 #0.015

    S = np.eye(N)
    l = S.flatten(order='F')
    sigma = np.eye(N**2)
    K = np.zeros((N, N))
    R = sigma_r**2 * np.eye(N)
    Q = sigma_q**2 * np.eye(N**2)
    M = 10
    T = 60
    alpha = 5

    least_squares_P = np.zeros((simulation_steps // T + 1, N))
    least_squares_Y = np.zeros((simulation_steps // T, N))

    last_CR = np.zeros(N)

    number_of_clicks = np.zeros(N)

    for i in range(1, simulation_steps + 1):
        opinions[i] = sim.update(p=position[i - 1])
        
        if(i % T == 0):
            least_squares_Y[i // T - 1] = (number_of_clicks / 40 - 0.5) / (0.5 * position[i - 1])
            delta_CR = number_of_clicks/40 - last_CR
            last_CR = number_of_clicks/40
            number_of_clicks = np.zeros(N)

            H = build_delta_cr(position[i - 1], position[i - 1 - T])
            R = sigma_r**2 * np.eye(N)
            
            K = sigma @ np.transpose(H) @ np.linalg.inv(R + H @ sigma @ np.transpose(H))
            l = l + K @ (delta_CR - (H @ l))
            
            Q = sigma_q**2 * np.eye(N**2)

            sigma = sigma + Q - K @ H @ sigma

            S = np.reshape(l, (N, N), order='F')

            sensitivity_error[i-1] = np.mean(np.abs(np.diag(sim.get_sensitivity()) - np.diag(S)) / np.abs(np.diag(sim.get_sensitivity()))) * 100

            position[i] = np.random.uniform(-1, 1, N)
            least_squares_P[i // T] = position[i]
        
        elif(i % T >= 20):
            position[i] = position[i - 1]
            number_of_clicks += (np.random.rand(N) < clicking_function(position[i], opinions[i])).astype(int)
            
        else:
            position[i] = position[i - 1]
    
    #print("original sensitivity: ", np.diag(sim.get_sensitivity()), "\n")

    mask =  np.all(np.isfinite(least_squares_Y), axis=1)
    least_squares_Y = least_squares_Y[mask]
    least_squares_P = least_squares_P[:-1, :][mask]
    S_least_squares = least_squares_Y.T @ np.linalg.pinv(least_squares_P.T)

    #print(np.diag(S))
    #print(np.diag(S_least_squares))


    sensitivity_error = sensitivity_error[sensitivity_error != 0]
    ls_error = np.mean(np.abs(np.diag(sim.get_sensitivity()) - np.diag(S_least_squares)) / np.abs(np.diag(sim.get_sensitivity()))) * 100

    plt.plot(sensitivity_error, linestyle='-', label='Kalman Filter Sensitivity Error')
    plt.plot(ls_error * np.ones(len(sensitivity_error)), linestyle='--', color='red', label='Least Squares Sensitivity Error')
    plt.ylim(0,200)
    plt.grid()
    plt.legend()
    return S, S_least_squares


# Parameters
simulation_steps = 6000
N = 5
trials = 50
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
    bias_kf_ls[t] = np.mean(np.diag(S) - np.diag(S_LS))

print(np.mean(bias_kf), bias_kf)
print(np.mean(bias_ls), bias_ls)
print(np.mean(bias_kf_ls), bias_kf_ls)

#plt.show()