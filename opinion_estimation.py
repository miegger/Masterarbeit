import numpy as np
import matplotlib.pyplot as plt
from helper.opinion_dynamics import FJModel, generate_adjacency_matrix
from scipy.special import expit
from scipy.optimize import minimize


def calculate_clicking_probability(x, p):
    return 1 - 0.25*(x-p)**2


# Parameters
simulation_steps = 100
N = 15
runs = 1

# Fixed matrix and parameters
A = generate_adjacency_matrix(N)
gamma_p = np.random.uniform(0.01, 0.5, N)
x_0 = np.random.uniform(-1, 1, N)
gamma_d = 0.5 * np.abs(x_0)

# To store clicks from all runs
clicks = np.zeros((simulation_steps, N))

sim = FJModel(N=N, gamma_p=gamma_p, gamma_d=gamma_d, A=A, x_0=x_0)

opinions = np.zeros((simulation_steps + 1, N))
opinions[0] = x_0

position = np.zeros((simulation_steps, N))


for i in range(simulation_steps):
    position[i] = np.random.randint(3,size=N) - 1
    clicks[i] = (np.random.rand(N) < calculate_clicking_probability(position[i], opinions[i])).astype(int)

    opinions[i + 1] = sim.update(p=position[i])

error = np.zeros((N))

for user in range(N):
    np.random.seed(42)

    # --- Parameters for synthetic data ---
    T = simulation_steps  # Number of time steps
    true_alpha = 5  # Sharpness of click probability
    true_beta = 0  # Bias term

    # --- Generate article positions ---
    p_t = position[:,user]

    # --- Generate true user opinion trajectory ---
    x_true = opinions[:,user]

    # --- Define click probability function ---
    def click_prob(x, p, alpha=true_alpha):
        return 1 - 0.25*(x-p)**2

    # --- Generate click data ---
    c_t = clicks[:,user]

    # --- Initialize EM estimation ---
    x_est = np.zeros(T)
    x_est[0] = np.random.uniform(-1, 1)
    eta = 0.05  # Initial guess for eta

    n_iters = 10  # Number of EM iterations

    for iteration in range(n_iters):
        # --- E-step: Update latent states x_est based on current eta ---
        for t in range(1, T):
            x_est[t] = x_est[t-1] + eta * (p_t[t-1] - x_est[t-1])

        # --- M-step: Optimize eta to maximize likelihood ---
        def neg_log_likelihood(eta_val):
            x_temp = np.zeros(T)
            x_temp[0] = x_est[0]
            log_likelihood = 0.0
            for t in range(1, T):
                x_temp[t] = x_temp[t-1] + eta_val * (p_t[t-1] - x_temp[t-1])
                prob = click_prob(x_temp[t], p_t[t])
                log_likelihood += c_t[t] * np.log(prob + 1e-9) + (1 - c_t[t]) * np.log(1 - prob + 1e-9)
            return -log_likelihood

        result = minimize(neg_log_likelihood, eta, bounds=[(0.0, 1.0)])
        eta = result.x[0]

    error[user] = np.mean(np.abs(x_true[0:-1]-x_est))
    print(f"Estimated eta: {eta:.4f}")

    # --- Plot the results ---
    plt.figure(figsize=(10, 5))
    plt.plot(x_true, label='True Opinion', linewidth=2)
    plt.plot(x_est, label='Estimated Opinion', linestyle='--', linewidth=2)
    plt.scatter(range(T), p_t, label='Article Positions', alpha=0.4)
    plt.scatter(range(T), c_t * 2 - 1, label='Clicks (1=click, 0=no)', c='red', marker='x')
    plt.title(f"EM Opinion Estimation (Estimated eta = {eta:.3f})")
    plt.xlabel("Time")
    plt.ylabel("Opinion / Position")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

print(error)