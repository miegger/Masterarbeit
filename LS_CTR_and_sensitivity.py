import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from helper.opinion_dynamics import DGModel

# --- Simulation settings ---
#np.random.seed(42)
num_samples = 50
d = 5  # Dimension of p and x
beta_true = 1.5
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
x_0 = np.random.uniform(-1, 1, d)
gamma_p = np.random.uniform(0.01, 0.5, d)
sim = DGModel(N=d, gamma=gamma_p, A=A, x_0=x_0)

# True W (unknown to learner)
W_true = sim.get_sensitivity()

# Generate random article positions (shape: num_samples x d)
P = np.random.uniform(low=-1, high=1, size=(num_samples, d))

def clicking_function(p, opinions, beta_true):
    return np.exp(-beta_true * (p - opinions)**2)

def clicking_function1(p, opinions):
    return 0.5 * np.ones_like(p) + 0.5 * p * opinions

def clicking_function2(p, opinions):
    return 0.25 * np.ones_like(p) - (p - opinions)**2

CTR_obs = np.zeros((num_samples, d))  # Observed CTRs
X = np.zeros((num_samples, d))

# Compute true x = Wp and CTRs
for i in range(num_samples):
    for j in range(20):
        sim.update(P[i])

    number_of_clicks = np.zeros(d)
    for j in range(40):
        x = sim.update(P[i])
        number_of_clicks += (np.random.rand(d) < clicking_function(P[i], x, beta_true)).astype(int)

    CTR_obs[i] = number_of_clicks / 40  # Average CTR over 40 trials
    X[i] = sim.get_opinion()

delta_p = np.diff(P[0:45], axis=0)
delta_cr = np.diff(CTR_obs[0:45], axis=0)

A = delta_cr.T @ delta_p @ np.linalg.inv(delta_p.T @ delta_p)


#print(CTR_obs[45:])
#print(A @ P[45:].T)

"""
sim2 = DGModel(N=d, gamma=gamma_p, A=A, x_0=x_0)
X2 = np.zeros((2,d))
P2 = P[:2]
CTR_obs2 = np.zeros((2, d))  # Observed CTRs for two users
for i in range(2):
    for j in range(20):
        sim2.update(P[i])

    number_of_clicks = np.zeros(d)
    for j in range(40):
        x = sim2.update(P[i])
        number_of_clicks += (np.random.rand(d) < clicking_function(P[i], x, beta_true)).astype(int)

    CTR_obs2[i] = number_of_clicks / 40  # Average CTR over 40 trials
    X2[i] = sim2.get_opinion()

S = sim2.get_sensitivity()

print(CTR_obs2[0])
print(CTR_obs2[1] + (0.5 * S @ P[1] + 0.5 * S.T @ P[1]) * (P[0] - P[1]))
"""

# --------------------------
# Estimation Function (with beta)
# --------------------------
def ctr_loss(params, P, CTR_obs, d):
    """
    params contains flattened W (d*d) followed by scalar beta.
    """
    W_flat = params[:d*d]
    beta = params[-1]
    if beta <= 0:
        return 1e10  # Penalize negative or zero beta to keep it positive
    
    W = W_flat.reshape((d, d))
    X_hat = P @ W.T
    D = X_hat - P
    CTR_pred = np.exp(-beta * D**2)
    return np.mean((CTR_pred - CTR_obs)**2)

# --------------------------
# Optimization
# --------------------------
W0 = np.random.randn(d, d) * 0.1
beta0 = 1.0  # Initial guess for beta
init_params = np.concatenate([W0.flatten(), [beta0]])

result = minimize(
    ctr_loss,
    init_params,
    args=(P, CTR_obs, d),
    method='L-BFGS-B',
    bounds=[(0, 1)]*(d*d) + [(1e-6, None)],  # beta constrained > 0
    options={'disp': True}
)
params_est = result.x
W_est = params_est[:d*d].reshape((d, d))
beta_est = params_est[-1]

print(f"Estimated beta: {beta_est:.4f}")

# --------------------------
# Evaluation
# --------------------------
X_est = P @ W_est.T
D_est = X_est - P
CTR_est = np.exp(-beta_est * D_est**2)

# Print estimation error
print("\nEstimation error (||W_true - W_est||_F):", np.mean(W_true - W_est)**2)

# Compare predicted and true CTRs for a few users
plt.figure(figsize=(10, 6))
for u in range(min(d, 3)):
    plt.plot(CTR_obs[:, u], label=f'Observed CTR (user {u})', alpha=0.5)
    plt.plot(CTR_est[:, u], '--', label=f'Predicted CTR (user {u})', alpha=0.8)
plt.xlabel('Time (Article Index)')
plt.ylabel('CTR')
plt.legend()
plt.title('Observed vs Predicted CTR (for selected users)')
plt.grid(True)
plt.tight_layout()
#plt.show()


