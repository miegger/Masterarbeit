import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from helper.opinion_dynamics import DGModel

# --- Simulation settings ---
np.random.seed(10)
num_samples = 5
d = 5  # Dimension of p and x
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
x_0 = np.random.uniform(-1, 1, d)
gamma_p = np.random.uniform(0.01, 0.5, d)
sim = DGModel(N=d, gamma=gamma_p, A=A, x_0=x_0)
theta = np.random.normal(0.25, 0.1, d)

# Generate random article positions (shape: num_samples x d)
P = np.random.uniform(low=-1, high=1, size=(num_samples, d))

def clicking_function(p, opinions, theta):
    """Clicking function model: 1 - theta * (opinions - p)**2"""
    return np.ones_like(p) - theta*(opinions - p)**2

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
        number_of_clicks += (np.random.rand(d) < clicking_function(P[i], x, theta)).astype(int)
    X[i] = sim.get_sensitivity() @ P[i]

    #CTR_obs[i] = number_of_clicks / 40  # Average CTR over 40 trials
    CTR_obs[i] = clicking_function(P[i], X[i], theta)  # Perfect clicking function
    
    true_sensitivity[i] = sim.get_G(P[i], theta)


def equations(vars, G, P, d):
    n_G = G.shape[0]  # number of G matrices
    x = vars[0 : n_G * d]
    theta = vars[n_G * d : n_G * d + d]
    S = vars[n_G * d + d :].reshape((d, d), order='F')

    eq = []
    
    for i in range(d):
        
        for g in range(n_G):
            eq.append(2 * theta[i] * (x[g * d + i] - P[g][i]) * (S[i][i] - 1) - G[g][i][i])  # diagonal terms
            
            steady_state = -1 * x[g * d + i]
            for j in range(d):
                steady_state += S[i][j] * P[g][j]
            eq.append(steady_state) # steady state equation x = S @ P[g]

        for j in range(d):
            if i != j:
                for g in range(n_G):
                    eq.append(2 * theta[i] * (x[g * d + i] - P[g][i]) * S[i][j] - G[g][i][j])  # off-diagonal terms

        eq.append(np.sum(S[i, :]) - 1)  # row sum of S equals 1

    return eq

def objective(vars, G, P, d):
    return np.sum(np.abs(np.array(equations(vars, G, P, d))))

def constraint1(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[1][1] - S[2][1]
def constraint2(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[1][1] - S[3][1]
def constraint3(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[1][1] - S[4][1]
def constraint4(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[1][1] - S[0][1]
def constraint5(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[4][4] - S[0][4]
def constraint6(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[4][4] - S[1][4]
def constraint7(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[4][4] - S[2][4]
def constraint8(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[4][4] - S[3][4]
def constraint9(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[0][0] - S[1][0]
def constraint10(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[0][0] - S[2][0]
def constraint11(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[0][0] - S[3][0]
def constraint12(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[0][0] - S[4][0]
def constraint13(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[2][2] - S[0][2]
def constraint14(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[2][2] - S[1][2]
def constraint15(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[2][2] - S[3][2]
def constraint16(vars):
    S = vars[num_samples * d + d :].reshape((d, d), order='F')
    return S[2][2] - S[4][2]

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
        {'type': 'ineq', 'fun': constraint12},
        {'type': 'ineq', 'fun': constraint13},
        {'type': 'ineq', 'fun': constraint14},
        {'type': 'ineq', 'fun': constraint15},
        {'type': 'ineq', 'fun': constraint16}]

num_vars = num_samples * d + d + d**2
lower_bound = np.concatenate((-1 * np.ones(num_samples * d), -np.inf * np.ones(d), np.zeros(d**2)))
upper_bound = np.concatenate((1 * np.ones(num_samples * d), np.inf * np.ones(d), np.ones(d**2)))
initial_guess = np.concatenate((np.zeros(num_samples * d + d), 0.5 * np.ones(d**2)))


result = minimize(
    objective,
    initial_guess,
    bounds=list(zip(lower_bound, upper_bound)),
    method='SLSQP',
    constraints=cons,
    args=(true_sensitivity, P, d),
    options={'disp': True, 'maxiter': 1000}
)
"""

result = least_squares(
    equations,
    initial_guess,
    ftol=1e-12,
    xtol=1e-12,
    gtol=1e-14,
    bounds=[lower_bound, upper_bound],
    args=(true_sensitivity, P, d),
)
"""
# Print results
print("Success:", result.success)
print("Message:", result.message)

for g in range(num_samples):
    print(f"True x[{g}]:", X[g])
    print(f"Estimated x[{g}]:", result.x[g*d:(g+1)*d])

print(f"True theta:", theta)
print(f"Estimated theta:", result.x[num_samples*d : num_samples*d + d])

S_estimated = result.x[num_samples * d + d:].reshape((d, d), order='F')
print("True sensitivity matrix:\n", np.round(sim.get_sensitivity(), 3))
print("Estimated sensitivity matrix:\n", np.round(S_estimated, 3))
print("True col sum:", np.round(np.sum(sim.get_sensitivity(), axis=0), 3))
print("Estimated col sum:", np.round(np.sum(S_estimated, axis=0), 3))

#print(result.fun)
#print(P[0])
#print(true_sensitivity[0])

#print("True equations:", np.array(equations(np.concatenate((X.flatten(), theta, sim.get_sensitivity().flatten(order='F'))), true_sensitivity, P, d)))
#print("Estimated equations:", np.array(equations(result.x, true_sensitivity, d)))