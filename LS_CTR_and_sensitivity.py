import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from helper.opinion_dynamics import DGModel

# --- Simulation settings ---
np.random.seed(13)
num_samples = 50
d = 5  # Dimension of p and x
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
x_0 = np.random.uniform(-1, 1, d)
gamma_p = np.random.uniform(0.01, 0.5, d)
sim = DGModel(N=d, gamma=gamma_p, A=A, x_0=x_0)
beta_true = np.random.normal(0.25, 0.1, d)

# True W (unknown to learner)
W_true = sim.get_sensitivity()

# Generate random article positions (shape: num_samples x d)
P = np.random.uniform(low=-1, high=1, size=(num_samples, d))

def clicking_function(p, opinions, beta_true):
    return np.exp(-beta_true * (p - opinions)**2)

def clicking_function1(p, opinions):
    return 0.5 * np.ones_like(p) + 0.5 * p * opinions

def clicking_function2(p, opinions, theta_true = 0.25):
    return np.ones_like(p) - theta_true * (p - opinions)**2

CTR_obs = np.zeros((num_samples, d))  # Observed CTRs
X = np.zeros((num_samples, d))

# Compute true x = Wp and CTRs
for i in range(num_samples):
    for j in range(20):
        sim.update(P[i])

    number_of_clicks = np.zeros(d)
    for j in range(40):
        x = sim.update(P[i])
        number_of_clicks += (np.random.rand(d) < clicking_function2(P[i], x, beta_true)).astype(int)

    #CTR_obs[i] = number_of_clicks / 40  # Average CTR over 40 trials
    CTR_obs[i] = clicking_function2(P[i], x, beta_true)
    X[i] = W_true @ P[i]


# --------------------------
# Estimation Function (with beta)
# --------------------------
def ctr_loss(params, P, CTR_obs, d):
    """
    params contains flattened W (d*d) followed by scalar beta.
    """
    W_flat = params[:d*d]
    theta = params[d*d:]
    
    W = W_flat.reshape((d, d))
    X_hat = P @ W.T
    D = X_hat - P
    CTR_pred = np.ones(d) - theta * (D)**2
    return np.mean((CTR_pred - CTR_obs)**2)

# --------------------------
# Optimization
# --------------------------
W0 = np.ones((d,d)) * 0.5
theta0 = np.zeros(d)  # Initial guess for theta
init_params = np.concatenate([W0.flatten(), theta0])

result = minimize(
    ctr_loss,
    init_params,
    args=(P, CTR_obs, d),
    method='L-BFGS-B',
    bounds=[(0, 1)]*(d*d) + [(0, None)] * d,
    options={'disp': True}
)
params_est = result.x
W_est = params_est[:d*d].reshape((d, d))
theta_est = params_est[d*d:]

print(f"True theta:", np.round(beta_true, 4))
print(f"Estimated theta:", np.round(theta_est, 4))

# --------------------------
# Evaluation
# --------------------------
X_est = P @ W_est.T
D_est = X_est - P
CTR_est = np.exp(-theta_est * D_est**2)

print("True sensitivity: \n", np.round(W_true, 3))
print("Estimated sensitivity: \n", np.round(W_est, 3))

# Print estimation error
print("\nEstimation error (||W_true - W_est||_F):", np.mean(W_true - W_est)**2)

# Compare predicted and true CTRs for a few users
plt.figure(figsize=(10, 6))
for u in range(min(d, 2)):
    plt.plot(CTR_obs[:, u], label=f'Observed CTR (user {u})', alpha=0.5)
    plt.plot(CTR_est[:, u], '--', label=f'Predicted CTR (user {u})', alpha=0.8)
plt.xlabel('Time (Article Index)')
plt.ylabel('CTR')
plt.legend()
plt.title('Observed vs Predicted CTR (for selected users)')
plt.grid(True)
plt.tight_layout()
#plt.show()