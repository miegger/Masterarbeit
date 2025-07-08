import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from helper.opinion_dynamics import DGModel

# --- Simulation settings ---
np.random.seed(39)
num_samples = 100
d = 5  # Dimension of p and x
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
x_0 = np.random.uniform(-1, 1, d)
gamma_p = np.random.uniform(0.01, 0.5, d)
sim = DGModel(N=d, gamma=gamma_p, A=A, x_0=x_0)

max_step = 0.05    # maximum allowed change per element

# Initialize the array
P = np.zeros((num_samples, d))

# Start from a random initial point in [-1, 1]
P[0] = np.random.uniform(low=-1, high=1, size=(d,))

# Random walk: each step changes slightly from the previous
for i in range(1, num_samples):
    step = np.random.uniform(low=-max_step, high=max_step, size=(d,))
    P[i] = P[i - 1] + step
    # Optional: Clip to stay in [-1, 1]
    P[i] = np.clip(P[i], -1, 1)


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
    CTR_obs[i] = clicking_function1(P[i], sim.get_opinion())  # Average CTR over 40 trials

    X[i] = sim.get_opinion()
    true_sensitivity[i] = sim.get_CR_sensitivity(P[i])

def kalman_filter(delta_p, delta_cr, true_sensitivity, N):
    sigma_r = 0.1 # Std of measurement noise --> low values we trust fully the CTR
    sigma_q = 0.1 # Std of process noise

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
        
        denominator = np.where(np.abs(true_sensitivity[i]) <= 1e-3, 1, np.abs(true_sensitivity[-1]))
        error[i] = 100*np.mean(np.abs(l.reshape((N, N),order='F') - true_sensitivity[i]) / denominator)

        #error[i] = np.linalg.norm(l.reshape((N, N), order='F') - true_sensitivity[i], ord='fro')
        #print(np.round(delta_cr[i], 3), np.round(true_sensitivity[i] @ delta_p[i], 3))
        #print(np.round(delta_cr[i], 3), np.round(l.reshape((N, N), order='F') @ delta_p[i], 3), "\n")

    return l.reshape((N, N), order='F'), error

sensitivity, err = kalman_filter(np.diff(P, axis=0), np.diff(CTR_obs, axis=0), true_sensitivity, d)
print(err)

print("Estimation:", np.round(sensitivity, 3))
print("True:", np.round(true_sensitivity[-1], 3))




###################################33

def equations(vars, G, d):
    n_G = 1  # number of G matrices
    x = vars[0 : n_G * d]
    y = vars[n_G * d : 2 * n_G * d]
    S = vars[2 * n_G * d :].reshape((d, d), order='F')

    eq = []

    for i in range(d):
        
        #for g in range(n_G):
        #    eq.append(x[g * d + i] + y[g * d + i] * S[i][i] - G[g][i][i])  # diagonal terms

        for j in range(d):
            if i != j:
                eq.append(y[i] * S[i][j] - G[i][j])  # off-diagonal terms

        eq.append(np.sum(S[i, :]) - 1)  # row sum of S equals 1

    return eq

def objective(vars, G, d):
    return np.sum(np.abs(np.array(equations(vars, G, d))))

def constraint1(vars):
    S = vars[2 * 1 * 5 :].reshape((d, d), order='F')
    return S[1][1] - S[2][1]
def constraint2(vars):
    S = vars[2 * 1 * 5 :].reshape((d, d), order='F')
    return S[1][1] - S[3][1]
def constraint3(vars):
    S = vars[2 * 1 * 5 :].reshape((d, d), order='F')
    return S[1][1] - S[4][1]
def constraint4(vars):
    S = vars[2 * 1 * 5 :].reshape((d, d), order='F')
    return S[1][1] - S[0][1]
def constraint5(vars):
    S = vars[2 * 1 * 5 :].reshape((d, d), order='F')
    return S[4][4] - S[0][4]
def constraint6(vars):
    S = vars[2 * 1 * 5 :].reshape((d, d), order='F')
    return S[4][4] - S[1][4]
def constraint7(vars):
    S = vars[2 * 1 * 5 :].reshape((d, d), order='F')
    return S[4][4] - S[2][4]
def constraint8(vars):
    S = vars[2 * 1 * 5 :].reshape((d, d), order='F')
    return S[4][4] - S[3][4]
def constraint9(vars):
    S = vars[2 * 1 * 5 :].reshape((d, d), order='F')
    return S[0][0] - S[1][0]
def constraint10(vars):
    S = vars[2 * 1 * 5 :].reshape((d, d), order='F')
    return S[0][0] - S[2][0]
def constraint11(vars):
    S = vars[2 * 1 * 5 :].reshape((d, d), order='F')
    return S[0][0] - S[3][0]
def constraint12(vars):
    S = vars[2 * 1 * 5 :].reshape((d, d), order='F')
    return S[0][0] - S[4][0]

cons = [{'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3},
        {'type': 'ineq', 'fun': constraint4},
        {'type': 'ineq', 'fun': constraint5},
        {'type': 'ineq', 'fun': constraint6},
        {'type': 'ineq', 'fun': constraint7},
        {'type': 'ineq', 'fun': constraint8},
        {'type': 'ineq', 'fun': constraint9},
        {'type': 'ineq', 'fun': constraint10},
        {'type': 'ineq', 'fun': constraint11},
        {'type': 'ineq', 'fun': constraint12},]

num_samples = 1

num_vars = 2 * num_samples * d + d**2
lower_bound = np.concatenate((-np.inf * np.ones(2 * num_samples * d), np.zeros(d**2)))
upper_bound = np.concatenate((np.inf * np.ones(2 * num_samples * d), np.ones(d**2)))
initial_guess = np.concatenate((np.zeros(2 * num_samples * d), 0.5 * np.ones(d**2)))

"""
result = minimize(
    objective,
    initial_guess,
    bounds=list(zip(lower_bound, upper_bound)),
    method='SLSQP',
    constraints=cons,
    args=(sensitivity, d),
)
"""

result = least_squares(
    equations,
    initial_guess,
    ftol=1e-12,
    xtol=1e-12,
    gtol=1e-16,
    bounds=[lower_bound, upper_bound],
    args=(sensitivity, d),
)

# Print results
print("Success:", result.success)
print("Message:", result.message)

for g in range(num_samples):
    print(f"True x[{g}]:", 0.5 * X[g])
    print(f"Estimated x[{g}]:", result.x[g*d:(g+1)*d])

for g in range(num_samples):
    print(f"True y[{g}]:", 0.5 * P[g])
    print(f"Estimated y[{g}]:", result.x[num_samples*d + g*d : num_samples*d + (g+1)*d])

S_estimated = result.x[2 * num_samples * d:].reshape((d, d), order='F')
print("True sensitivity matrix:\n", np.round(sim.get_sensitivity(), 3))
print("Estimated sensitivity matrix:\n", np.round(S_estimated, 3))
print("True col sum:", np.round(np.sum(sim.get_sensitivity(), axis=0), 3))
print("Estimated col sum:", np.round(np.sum(S_estimated, axis=0), 3))

print(result.fun)


