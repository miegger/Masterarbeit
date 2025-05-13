import matplotlib.pyplot as plt
import numpy as np
from helper.opinion_dynamics import DGModel
import cvxpy as cp

# Parameters
simulation_steps = 50
N = 5
trials = 50

# Fixed matrix and parameters
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])

# To store positions from all runs
sensitivity_error = np.zeros((trials, simulation_steps))
sensitivity_error_diag = np.zeros((trials, simulation_steps))


for t in range(trials):
    x_0 = np.random.uniform(-1, 1, N)
    gamma_p = np.random.uniform(0.01, 0.5, N)

    sim = DGModel(N=N, gamma=gamma_p, A=A, x_0=x_0)
    true_sensitivity = sim.get_sensitivity()
    
    opinions = np.zeros((simulation_steps + 1, N))
    opinions[0] = x_0
    position = np.zeros((simulation_steps + 1, N))

    # Parameters used for the sensitivity estimation
    sigma_r = 0.1
    sigma_q = 0.1

    H = np.eye(N)
    l = H.flatten(order='F')
    sigma = np.eye(N**2)
    K = np.zeros((N, N))
    R = sigma_r**2 * np.eye(N)
    Q = sigma_q**2 * np.eye(N**2)
    M = 10
    
    delta_x = np.zeros(N)
    delta_p = np.zeros(N)

    

    for i in range(simulation_steps):
        opinions[i + 1] = sim.update(p=position[i])
        delta_x = opinions[i + 1] - opinions[i]

        kroeneker = np.kron(np.transpose(delta_p), np.eye(N))
        
        # Sanjay's method on predicting sigma_r
        sigma_r = np.sqrt(np.mean((delta_x - kroeneker @ l) ** 2))
        #print(sigma_mle)

        #n = np.linalg.norm(delta_p, ord=2)
        #if n != 0:
        #    sigma_r = 1.5 * n
        R = sigma_r**2 * np.eye(N)

        K = sigma @ np.transpose(kroeneker) @ np.linalg.inv(R + kroeneker @ sigma @ np.transpose(kroeneker))
        l = l + K @ (delta_x - kroeneker @ l)
        #l[l < 0] = 0

        # Sanjay's mehtod on predicting sigma_q
        non_zero_values = l[l != 0]
        min_non_zero = np.min(non_zero_values)
        sigma_q = min_non_zero / M
        
        Q = sigma_q**2 * np.eye(N**2)

        sigma = sigma + Q - K @ kroeneker @ sigma

        # Constrained Kalman filter
        """
        x = cp.Variable(N**2)
        objective = cp.Minimize(cp.quad_form(x - l, np.linalg.inv(sigma)))
        constraints = [x >= 0]
        for row in range(N):
            indices = [row + col * N for col in range(N)]
            constraints.append(cp.sum(x[indices]) == 1)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        l = x.value
        """
        H = np.reshape(l, (N, N), order='F')

        num = np.linalg.norm(true_sensitivity - H, ord='fro')
        denom = np.linalg.norm(true_sensitivity, ord='fro')
        sensitivity_error[t][i] = num / denom

        num_diag = np.linalg.norm(np.diag(true_sensitivity)- np.diag(H), ord=2)
        denom_diag = np.linalg.norm(np.diag(true_sensitivity), ord=2)
        sensitivity_error_diag[t][i] = num_diag / denom_diag

        position[i + 1] = sim.ofo_sensitivity(prev_p=position[i], sensitivity=H)
        delta_p = position[i + 1] - position[i]


print("True sensitivity: \n", np.sum(true_sensitivity, axis = 0))
print("H", np.sum(H, axis = 0))
fig, axes = plt.subplots(nrows=1, ncols=2)

mean_run = np.mean(sensitivity_error, axis=0)
print("Mean error all values: ", mean_run[-1])
for trial in sensitivity_error:
    axes[0].plot(trial, color='lightgrey', linewidth=1)  # Each row is one line
axes[0].plot(mean_run, color='red', linewidth=2.5, label='Average')
axes[0].set_title("All values")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Relative error")
axes[0].set_ylim(0.3, 1.4)
axes[0].legend()
axes[0].grid(True)

mean_run = np.mean(sensitivity_error_diag, axis=0)
print("Mean error diag: ", mean_run[-1])
for trial in sensitivity_error_diag:
    axes[1].plot(trial, color='lightgrey', linewidth=1)  # Each row is one line
axes[1].plot(mean_run, color='red', linewidth=2.5, label='Average')
axes[1].set_title("Diagonal values")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Relative error")
axes[1].set_ylim(0.2, 1.8)
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Relative error of the estimated sensitivity")
plt.show()

"""
print(all_positions)
# Create the boxplot
plt.figure(figsize=(8, 5))
plt.boxplot(all_positions, vert=True, patch_artist=True, showfliers=False)

# Labeling
plt.ylabel("Position")
plt.title("Boxplot of positions from 50 trials, constraint on ||p||_1 <= 2")
plt.xticks(range(1, N + 1), [f'Col {i}' for i in range(1, N + 1)])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot(all_norms, positions=[1], vert=True, patch_artist=True, showfliers=False)
plt.boxplot(all_norms_comparision, positions = [1.4], vert=True, patch_artist=True, showfliers=False)

# Labeling
plt.ylabel("Position")
plt.title("Cost function OFO vs [0,1,0,0,1]")
plt.xticks([1, 1.4], ['OFO', '[0,1,0,0,1]'])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""