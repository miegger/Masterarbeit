import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from helper.opinion_dynamics import DGModel

# --- Simulation settings ---
np.random.seed(39)
num_samples = 50
d = 5  # Dimension of p and x
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
x_0 = np.random.uniform(-1, 1, d)
gamma_p = np.random.uniform(0.01, 0.5, d)
sim = DGModel(N=d, gamma=gamma_p, A=A, x_0=x_0)

# Generate random article positions (shape: num_samples x d)
P = np.random.uniform(low=-1, high=1, size=(num_samples, d))
#P_1d = np.linspace(-1, 1, num_samples)
#P = np.tile(P_1d[:, None], (1, d))

def clicking_function(p, opinions, beta_true):
    return np.exp(-beta_true * (p - opinions)**2)

def clicking_function1(p, opinions):
    return 0.5 * np.ones_like(p) + 0.5 * p * opinions

def clicking_function2(p, opinions):
    return np.ones_like(p) - 0.25*(p - opinions)**2

CTR_obs = np.zeros((num_samples, d))  # Observed CTRs
X = np.zeros((num_samples, d))
true_sensitivity = np.zeros((num_samples, d, d))

# Compute true x = Wp and CTRs
for i in range(num_samples):
    for j in range(20):
        sim.update(P[i])

    number_of_clicks = np.zeros(d)
    for j in range(40):
        x = sim.update(P[i])
        number_of_clicks += (np.random.rand(d) < clicking_function1(P[i], x)).astype(int)

    CTR_obs[i] = number_of_clicks / 40  # Average CTR over 40 trials
    #CTR_obs[i] = clicking_function1(P[i], sim.get_opinion())  # Average CTR over 40 trials

    X[i] = sim.get_opinion()
    true_sensitivity[i] = sim.get_CR_sensitivity(P[i])

def kalman_filter(delta_p, delta_cr, true_sensitivity, N):
    sigma_r = 10 # Std of measurement noise --> low values we trust fully the CTR
    sigma_q = 1 # Std of process noise

    H = np.eye(N)
    l = H.flatten(order='F')
    sigma = np.eye(N**2)
    R = sigma_r**2 * np.eye(N)
    Q = sigma_q**2 * np.eye(N**2)

    error = np.zeros(len(delta_p))

    for i in range(len(delta_p)):
        kroeneker = np.kron(np.transpose(delta_p[i]), np.eye(N))
        K = sigma @ np.transpose(kroeneker) @ np.linalg.inv(R + kroeneker @ sigma @ np.transpose(kroeneker))
        l = l + K @ (delta_cr[i] - kroeneker @ l)

        sigma = sigma + Q - K @ kroeneker @ sigma

        error[i] = np.linalg.norm(l.reshape((N, N), order='F') - true_sensitivity[i], ord='fro')
        #print(np.round(delta_cr[i], 3), np.round(true_sensitivity[i] @ delta_p[i], 3))
        #print(np.round(delta_cr[i], 3), np.round(l.reshape((N, N), order='F') @ delta_p[i], 3), "\n")

    return error

#print(kalman_filter(np.diff(P, axis=0), np.diff(CTR_obs, axis=0), true_sensitivity, d))

print(np.round(true_sensitivity[0:5], 3))

