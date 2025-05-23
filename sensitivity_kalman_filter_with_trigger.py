import matplotlib.pyplot as plt
import numpy as np
from helper.opinion_dynamics import DGModel
import cvxpy as cp

def estimate_sensitivity(mode, sim, simulation_steps, true_sensitivity, opinions, position):
    """
    Estimates the sensitvity with a Kalman filter

    :param int mode: 1: constrained KF, 2: unconstrained KF
                    3: sigma_q from Sanjay, 4: unconstrained KF with sigma_q and sigma_r from Sanjay
                    5: sigma_q from Sanjay and sigma_r relative to delta_p, 6: sigma_q and sigma_r from Sanjay
                    7: naiv and QF 
    :return: 
    """
    sensitivity_error = np.zeros(simulation_steps)
    sensitivity_error_diag = np.zeros(simulation_steps)
    col_sum_error = np.zeros(simulation_steps) 

    sigma_r = 0.1
    sigma_q = 0.1

    H = np.eye(N)
    l = H.flatten(order='F')
    sigma = np.eye(N**2)
    K = np.zeros((N, N))
    R = sigma_r**2 * np.eye(N)
    Q = sigma_q**2 * np.eye(N**2)
    M = 10
    T = 20

    delta_x = np.zeros(N)
    delta_p = np.zeros(N)

    for i in range(1, simulation_steps + 1):
        opinions[i] = sim.update(p=position[i - 1])
        
        if(i % T == 0):
            delta_x = opinions[i] - opinions[i - T]

            kroeneker = np.kron(np.transpose(delta_p), np.eye(N))
            
            # Sanjay's method on predicting sigma_r
            if mode == 4 and mode == 6:
                sigma_r = np.sqrt(np.mean((delta_x - kroeneker @ l) ** 2))

            if mode == 5 and mode == 7:
                n = np.linalg.norm(delta_p, ord=2)**2
                if n != 0:
                    sigma_r = n

            R = sigma_r**2 * np.eye(N)

            K = sigma @ np.transpose(kroeneker) @ np.linalg.inv(R + kroeneker @ sigma @ np.transpose(kroeneker))
            l = l + K @ (delta_x - kroeneker @ l)

            # Sanjay's mehtod on predicting sigma_q
            if mode == 3 or mode == 4 or mode == 5 or mode == 6:
                non_zero_values = l[l != 0]
                min_non_zero = np.min(non_zero_values)
                sigma_q = min_non_zero / M
            
            Q = sigma_q**2 * np.eye(N**2)

            sigma = sigma + Q - K @ kroeneker @ sigma

            # Constrained Kalman filter
            if mode != 2 and mode != 4 and mode != 6:
                x = cp.Variable(N**2)
                objective = cp.Minimize(cp.quad_form(x - l, np.linalg.inv(sigma)))
                constraints = [x >= 0, x <= 1]
                if mode == 7:
                    for row in range(N):
                        indices = [row + col * N for col in range(N)]
                        constraints.append(cp.sum(x[indices]) == 1)
                problem = cp.Problem(objective, constraints)
                problem.solve()

                l = x.value

        H = np.reshape(l, (N, N), order='F')
 
        sensitivity_error[i - 1] = np.sum(np.abs(np.diag(true_sensitivity) - np.diag(H))) / np.sum(np.diag(true_sensitivity)) * 100
        sensitivity_error_diag[i - 1] = np.mean(np.abs(np.diag(true_sensitivity) - np.diag(H)) / np.abs(np.diag(true_sensitivity))) * 100
        col_sum_error[i - 1] = np.sum(np.abs(np.sum(true_sensitivity, axis=0) - np.sum(H, axis=0))) / np.sum(np.sum(true_sensitivity, axis=0)) * 100

        position[i] = position[i - 1]

        if(i % T == 0):
            position[i] = sim.ofo_sensitivity(prev_p=position[i - 1], sensitivity=H)
            delta_p = position[i] - position[i - 1]
            #print(position)
            #print(delta_p)
        else:
            position[i] = position[i - 1]

    #plt.plot(position)
    #plt.show()

    return sensitivity_error, sensitivity_error_diag, col_sum_error

# Parameters
simulation_steps = 1000
N = 5
trials = 50
modes = np.array([1,2,7])

# Fixed matrix and parameters
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])

sensitivity_error = np.zeros((modes.size, trials))
sensitivity_error_diag = np.zeros((modes.size, trials))
col_sum_error = np.zeros((modes.size, trials))

for m, mode in enumerate(modes):
    print("Mode: ", mode)
    s_err_temp = np.zeros((trials, simulation_steps))
    s_err_diag_temp = np.zeros((trials, simulation_steps))
    col_err_temp = np.zeros((trials, simulation_steps))
    for t in range(trials):
        x_0 = np.random.uniform(-1, 1, N)
        gamma_p = np.random.uniform(0.01, 0.5, N)

        sim = DGModel(N=N, gamma=gamma_p, A=A, x_0=x_0)
        true_sensitivity = sim.get_sensitivity()
        
        opinions = np.zeros((simulation_steps + 1, N))
        opinions[0] = x_0
        position = np.zeros((simulation_steps + 1, N))

        s_err_temp[t], s_err_diag_temp[t], col_err_temp[t] = estimate_sensitivity(mode=mode, sim=sim, simulation_steps=simulation_steps, true_sensitivity=true_sensitivity, opinions=opinions, position=position)
        sensitivity_error[m][t] = s_err_temp[t][-1]
        sensitivity_error_diag[m][t] = s_err_diag_temp[t][-1]
        col_sum_error[m][t] = col_err_temp[t][-1]
    
    plt.figure(m)

    fig, axes = plt.subplots(nrows=1, ncols=3)
    mean_run = np.mean(s_err_temp, axis=0)
    print("Mean error all values: ", mean_run[-1])
    for trial in s_err_temp:
        axes[0].plot(trial, color='lightgrey', linewidth=1)  # Each row is one line
    axes[0].plot(mean_run, color='red', linewidth=2.5, label='Average')
    axes[0].set_title("Overall Relative Error")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Relative error")
    axes[0].set_ylim(0, 200)
    axes[0].legend()
    axes[0].grid(True)

    mean_run = np.mean(s_err_diag_temp, axis=0)
    print("Mean error diag: ", mean_run[-1])
    for trial in s_err_diag_temp:
        axes[1].plot(trial, color='lightgrey', linewidth=1)  # Each row is one line
    axes[1].plot(mean_run, color='red', linewidth=2.5, label='Average')
    axes[1].set_title("Mean Relative Error")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Relative error")
    axes[1].set_ylim(0, 450)
    axes[1].legend()
    axes[1].grid(True)

    mean_run = np.mean(col_err_temp, axis=0)
    print("Mean error all values: ", mean_run[-1])
    for trial in col_err_temp:
        axes[2].plot(trial, color='lightgrey', linewidth=1)  # Each row is one line
    axes[2].plot(mean_run, color='red', linewidth=2.5, label='Average')
    axes[2].set_title("Overall Relative Error Column Sums")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Relative error")
    axes[2].set_ylim(0, 100)
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle(f"Relative error of the estimated sensitivity for mode {mode}")

# Create the boxplot
plt.figure(8)
plt.boxplot(sensitivity_error.T, vert=True, patch_artist=True, showfliers=False)

# Add labels (optional)
size = modes.size
plt.xticks(ticks=range(1, size + 1), labels=[f'Mode {modes[i]}' for i in range(0, size)])
plt.title("Overall relative error of the estimated sensitivity diagonal elements (T=20)")
plt.ylabel("Error")
plt.grid()
#plt.ylim(30,140)

# Create the boxplot
plt.figure(9)
plt.boxplot(sensitivity_error_diag.T, vert=True, patch_artist=True, showfliers=False)

# Add labels (optional)
plt.xticks(ticks=range(1, size + 1), labels=[f'Mode {modes[i]}' for i in range(0, size)])
plt.title("Mean relative error of the estimated sensitivity diagonal elements (T=20)")
plt.ylabel("Error")
plt.grid()
# Create the boxplot
plt.figure(10)
plt.boxplot(col_sum_error.T, vert=True, patch_artist=True, showfliers=False)

# Add labels (optional)
plt.xticks(ticks=range(1, size + 1), labels=[f'Mode {modes[i]}' for i in range(0, size)])
plt.title("Relative error of the estimated sensitivity column sums (T=20)")
plt.ylabel("Error")
plt.grid()
#plt.ylim(15,145)

plt.show()
"""
# Parameters
simulation_steps = 500
N = 5
trials = 50
T = 10

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

    H = np.eye(N, N)
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
        
        if(i % T == 0):
            delta_x = opinions[i + 1] - opinions[i - T]

            kroeneker = np.kron(np.transpose(delta_p), np.eye(N))
            
            # Sanjay's method on predicting sigma_r
            #sigma_r = np.sqrt(np.mean((delta_x - kroeneker @ l) ** 2))

            #n = np.linalg.norm(delta_p, ord=2)
            #if n != 0:
            #    sigma_r = 1.5 * n
            R = sigma_r**2 * np.eye(N)

            K = sigma @ np.transpose(kroeneker) @ np.linalg.inv(R + kroeneker @ sigma @ np.transpose(kroeneker))

            l = l + K @ (delta_x - kroeneker @ l)
        
            # Forcing the values of the sensitivity to non negative
            #l[l < 0] = 0

            # Sanjay's mehtod on predicting sigma_q

            #non_zero_values = l[l != 0]
            #min_non_zero = np.min(non_zero_values)
            #sigma_q = min_non_zero / M

            
            Q = sigma_q**2 * np.eye(N**2)

            sigma = sigma + Q - K @ kroeneker @ sigma

            # Constrained Kalman filter
            #
            x = cp.Variable(N**2)
            objective = cp.Minimize(cp.quad_form(x - l, np.linalg.inv(sigma)))
            constraints = [x >= 0]
            for row in range(N):
                indices = [row + col * N for col in range(N)]
                constraints.append(cp.sum(x[indices]) == 1)
            problem = cp.Problem(objective, constraints)
            problem.solve()

            l = x.value
            #
            H = np.reshape(l, (N, N), order='F')

        num = np.linalg.norm(true_sensitivity - H, ord='fro')
        denom = np.linalg.norm(true_sensitivity, ord='fro')
        sensitivity_error[t][i] = num / denom

        num_diag = np.linalg.norm(np.diag(true_sensitivity)- np.diag(H), ord=2)
        denom_diag = np.linalg.norm(np.diag(true_sensitivity), ord=2)
        sensitivity_error_diag[t][i] = num_diag / denom_diag

        position[i + 1] = position[i]

        if(i % T == 0):
            position[i + 1] = sim.ofo_sensitivity(prev_p=position[i], sensitivity=H)
            delta_p = position[i + 1] - position[i - T]

#print("True sensitivity: \n", true_sensitivity)
#print("H", H)
fig, axes = plt.subplots(nrows=1, ncols=2)

mean_run = np.mean(sensitivity_error, axis=0)
print("Mean error all values: ", mean_run[-1])
for trial in sensitivity_error:
    axes[0].plot(trial, color='lightgrey', linewidth=1)  # Each row is one line
axes[0].plot(mean_run, color='red', linewidth=2.5, label='Average')
axes[0].set_title("All values")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Relative error")
axes[0].set_ylim(0.3, 3.5)
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
axes[1].set_ylim(0.2, 2.5)
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Relative error of the estimated sensitivity with trigger")
plt.show()

plt.plot(position)
plt.show()

plt.plot(opinions)
plt.show()


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