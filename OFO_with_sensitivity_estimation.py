import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize
from helper.opinion_dynamics import DGModel, FJModel
from helper.clicking_functions import clicking_function_combined, minimize_combined, minimize_combined_scalar
import pandas as pd

# Parameters
#np.random.seed(41)
np.set_printoptions(precision=4, suppress=True)
simulation_steps = 5000
num_of_triggers = simulation_steps//100
d = 5
kappa = 2 # L1 constraint on P
const_A = True
flip_A = False

records = []

runs = 50
for run in range(runs):
    print("Run", run + 1)

    # Fixed matrix and parameters
    A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
    x_0 = np.random.uniform(-1, 1, d)
    p_0 = np.random.uniform(-1, 1, d)
    p_0 = (p_0 / np.linalg.norm(p_0, 1)) * kappa
    gamma_p = np.random.uniform(0.01, 0.5, d)
    gamma_d = np.random.uniform(0.01, 0.25, d)
    #sim = DGModel(N=d, gamma=gamma_p, A=A, x_0=x_0)
    #sim_ideal = DGModel(N=d, gamma=gamma_p, A=A, x_0=x_0)
    disturb = np.random.normal(0, 0.2, d)
    sim = FJModel(N=d, gamma_p=gamma_p, gamma_d=gamma_d, A=A, x_0=x_0, d=disturb)
    sim_ideal = FJModel(N=d, gamma_p=gamma_p, gamma_d=gamma_d, A=A, x_0=x_0, d=disturb)
    theta = np.random.uniform(0, 1, d)

    print("True theta:", theta)
    print("True sensitivity:\n", sim.get_sensitivity())
    #print("True row sums: \n", sim.get_sensitivity().sum(axis=1))

    # Storage for results
    CTR_obs = np.zeros((num_of_triggers, d))  # Observed CTRs
    CTR_obs_ideal = np.zeros((num_of_triggers, d))  # Observed CTRs
    X = np.zeros((simulation_steps + 1, d))
    X_ideal = np.zeros((simulation_steps + 1, d))
    X[0] = x_0
    X_ideal[0] = x_0
    P = np.zeros((num_of_triggers + 1, d))
    P_ideal = np.zeros((num_of_triggers + 1, d))
    P[0] = p_0
    P_ideal[0] = p_0

    sensitivity_error = np.zeros(num_of_triggers)
    rel_sensitivity_error_diag = np.zeros(num_of_triggers)
    theta_error = np.zeros(num_of_triggers)
    rel_theta_error = np.zeros(num_of_triggers)

    cost_difference = np.zeros(simulation_steps)

    estimated_theta = 0.5*np.ones(d)
    estimated_sensitivity = np.eye(d).flatten(order='C')

    trigger = 0
    start_range = 0
    for i in range(simulation_steps):
        cost_difference[i] = abs(np.sum(X[i]) - np.sum(X_ideal[i]))

        if i % 100 <= 40:  # First 20 steps: give time to converge
            X[i + 1] = sim.update(p=P[trigger])
            X_ideal[i + 1] = sim_ideal.update(p=P_ideal[trigger])
        else: # Last 60 steps: measure CTR
            CTR_obs[trigger] += np.random.rand() < clicking_function_combined(P[trigger], X[i], theta=theta)
            CTR_obs_ideal[trigger] += np.random.rand() < clicking_function_combined(P_ideal[trigger], X_ideal[i], theta=theta)
            X[i + 1] = sim.update(p=P[trigger])
            X_ideal[i + 1] = sim_ideal.update(p=P_ideal[trigger])


            if i % 100 == 99: # End of trigger period: average CTR, estimate sensitivity and choose new P
                #print("Trigger", trigger + 1)
                CTR_obs[trigger] /= 60
                CTR_obs_ideal[trigger] /= 60
                #print("Observed CTR:", CTR_obs[trigger])
                #print("Ideal CTR:", clicking_function_combined(P[trigger], X[i], theta=theta))

                if not const_A:
                    start_range = max(0, trigger - 50)

                #result = least_squares(minimize_combined, np.concatenate((np.zeros(d*d), np.ones(d))), bounds=(np.zeros(d*d + d), (np.ones(d*d + d))), verbose=0, args=(d, P[start_range:trigger+1], CTR_obs[start_range:trigger+1])).x
                
                #row-substochasticity constraints
                #"""
                constraints = [
                    {'type': 'ineq',
                    'fun': lambda params, i=i: 1 - np.sum(params[i*d:(i+1)*d])}
                    for i in range(d)
                ]
                #"""

                #row-stochasticity constraints for DG model!
                """
                constraints = [
                    {'type': 'eq', 'fun': lambda params, i=i: np.sum(params[i*d:(i+1)*d]) - 1}
                    for i in range(d)
                ]
                """

                # diagonal dominance constraints: S_ii >= S_ji for all j != i
                for i in range(d):  # for each column i
                    diag_idx = i * d + i
                    for j in range(d):
                        if j != i:
                            off_idx = j * d + i
                            constraints.append({
                                'type': 'ineq',
                                'fun': lambda params, diag_idx=diag_idx, off_idx=off_idx: params[diag_idx] - params[off_idx]
                            })
                
                result = minimize(minimize_combined_scalar, np.concatenate((estimated_sensitivity, estimated_theta)), bounds=[(0, 1)]*(d*d + d), constraints=constraints, args=(d, P[start_range:trigger+1], CTR_obs[start_range:trigger+1]), method='SLSQP', options={'maxiter': 1000})
                #print("Optimization success:", result.success, "Message:", result.message)
                result = result.x
                
                estimated_sensitivity = result[:d*d]
                estimated_sensitivity[np.abs(estimated_sensitivity) < 1e-10] = 0
                estimated_sensitivity_matrix = estimated_sensitivity.reshape((d, d), order='C')
                estimated_theta = result[d*d:]

                sensitivity_error[trigger] = np.linalg.norm(estimated_sensitivity_matrix - sim.get_sensitivity(), ord='fro')
                rel_sensitivity_error_diag[trigger] = 100 * np.linalg.norm(np.diag(estimated_sensitivity_matrix - sim.get_sensitivity()), ord=1) / np.linalg.norm(np.diag(sim.get_sensitivity()), ord=1)
                theta_error[trigger] = np.linalg.norm(estimated_theta - theta, ord=2)
                rel_theta_error[trigger] = 100 * np.linalg.norm(estimated_theta - theta, ord=1) / np.linalg.norm(theta, ord=1)

                #if trigger < 5:
                    #print("Estimated theta:", estimated_theta)
                    #print("Estimated sensitivity:\n", estimated_sensitivity)
                    #print("Estimated col sums: \n", estimated_sensitivity.reshape((d, d), order='C').sum(axis=0))

                P[trigger + 1] = sim.ofo_sensitivity(prev_p=P[trigger], sensitivity=estimated_sensitivity_matrix, constraint=kappa)
                P_ideal[trigger + 1] = sim_ideal.ofo(prev_p=P_ideal[trigger], constraint=kappa)
                trigger += 1
        
        if not const_A:
            sim.evolve_A()

        if i == simulation_steps//2 and flip_A:
            sim.A[[1,4]] = sim.A[[4,1]]

    #print("Estimated theta:", estimated_theta)
    #print("Estimated sensitivity:\n", estimated_sensitivity_matrix)

    records.append({"rel_sensitivity_error": rel_sensitivity_error_diag, "rel_theta_error": rel_theta_error, "cost":cost_difference, "positions": P, "opinions": X, "CTR": CTR_obs, "last_sensitivity_estimate": estimated_sensitivity_matrix, "last_theta_estimate": estimated_theta, "positions_ideal": P_ideal, "opinions_ideal": X_ideal, "CTR_ideal": CTR_obs_ideal})


df = pd.DataFrame(records)


def plot_results(l, title, values = 50):
    if len(l[0]) == 50:
        x = np.arange(50) * 100
    else:
        x = np.arange(len(l[0])) 
    
    for i in range(values):
        plt.plot(x, l[i], color='0.8')
    
    plt.plot(x, np.mean(l, axis=0), linestyle="-", label='Mean sensitivity error')
    plt.title(title)
    plt.xlabel("Time steps")
    plt.legend()
    plt.grid()
    plt.show()

plot_results(df['rel_sensitivity_error'].tolist(), title="Relative errror in %", values=runs)
plot_results(df['rel_theta_error'].tolist(), title=r"Relative $\theta$ error in %", values=runs)
plot_results(df['cost'].tolist(), title=r"Difference in average opinion", values=runs)



def plot_all_users(l, l_ideal, title):
    if len(l) == 50:
        x = np.arange(50) * 100
    else:
        x = np.arange(len(l[0])) 
    
    labels = [f"User {i + 1}" for i in range(5)]
    labels_ideal = [f"User {i + 1} (known sensitivity)" for i in range(5)]
    colors = ['b', 'g', 'r', 'c', 'm']
    
    l = list(zip(*l))
    l_ideal = list(zip(*l_ideal))

    for i in range(5):
        plt.plot(l[i], linestyle="-", label=labels[i], color=colors[i])
    
    for i in range(5):
        plt.plot(l_ideal[i], linestyle="--", label=labels_ideal[i], color=colors[i])
    
    plt.title(title)
    plt.xlabel("Time steps")
    plt.legend()
    plt.grid()
    plt.show()

#plot_all_users(df['positions'].tolist()[0], df['positions_ideal'].tolist()[0], title="Positions over time")
#plot_all_users(df['opinions'].tolist()[0], df['opinions_ideal'].tolist()[0], title="Opinions over time")


"""
print(P[-1])
print(P_ideal[-1])


plt.plot(sensitivity_error, label='Sensitivity error')
plt.plot(theta_error, label='Theta error')
plt.legend()
plt.grid()
plt.show()


plt.plot(rel_sensitivity_error_diag, label='Sensitivity error')
plt.plot(rel_theta_error, label='Theta error')
plt.title("Relative errror in %")
plt.xlabel("Time step")
plt.legend()
plt.grid()
plt.show()


plt.plot(np.sum(X, axis=1), label='Estimated sensitivity')
plt.plot(np.sum(X_ideal, axis=1), label='True sensitivity')

plt.xlabel("Time step")
plt.ylabel("Sum of opinions")

plt.grid()
plt.legend()
plt.show()
"""


"""
print("Constraint None: position", position[-1], "cost function", 0.5 * np.linalg.norm((opinions[-1] - np.ones(N))**2, ord=2))
fig, axs = plt.subplots(3, 1, figsize=(24, 8), sharex=True)

   
# Plot 1: Opinion Dynamics (OFO)
axs[0].plot(opinions)
axs[0].set_ylabel("Opinions")
axs[0].set_title("Opinion Dynamics (OFO, all users)")
axs[0].grid(True)
axs[0].legend([f"User {i+1}" for i in range(N)])
axs[0].set_ylim(-1, 1.05)
axs[0].set_xlim(0, simulation_steps)

ax2[0].plot(position)



sim = DGModel(N=N, gamma=gamma_p, A=A, x_0=x_0)

opinions = np.zeros((simulation_steps + 1, N))
opinions[0] = x_0

position = np.zeros((simulation_steps + 1, N))
delta = np.zeros((simulation_steps + 1, N))
delta[0] = np.ones(N)

for i in range(simulation_steps):
    opinions[i + 1] = sim.update(p=position[i])
    position[i + 1] = sim.ofo(prev_p=position[i], constraint=2)
    
print("Constraint 2: position", position[-1], "cost function", 0.5 * np.linalg.norm(opinions[-1] - np.ones(N), ord=2)**2)


# Plot 1: Opinion Dynamics (OFO)
axs[1].plot(opinions)
axs[1].set_ylabel("Opinions")
axs[1].set_title("Opinion Dynamics (OFO, 2 users)")
axs[1].grid(True)
axs[1].set_ylim(-1, 1.05)
axs[1].set_xlim(0, simulation_steps)

ax2[1].plot(position)


sim = DGModel(N=N, gamma=gamma_p, A=A, x_0=x_0)

opinions = np.zeros((simulation_steps + 1, N))
opinions[0] = x_0

position = np.zeros((simulation_steps + 1, N))
delta = np.zeros((simulation_steps + 1, N))
delta[0] = np.ones(N)

for i in range(simulation_steps):
    opinions[i + 1] = sim.update(p=np.array([0, 1, 0, 0, 1]))

print("Constraint 2 optimal: position", np.array([0, 1, 0, 0, 1]), "cost function", 0.5 * np.linalg.norm(opinions[-1] - np.ones(N), ord=2)**2)


# Plot 1: Opinion Dynamics (OFO)
axs[2].plot(opinions)
axs[2].set_ylabel("Opinions")
axs[2].set_title("Opinion Dynamics (2 users, ideal)")
axs[2].grid(True)
axs[2].set_ylim(-1, 1.05)
axs[2].set_xlim(0, simulation_steps)



# Plot 2: Opinion Dynamics (Feedforward)
axs[3].plot(opinions_all)
axs[3].set_xlabel("Time Step")
axs[3].set_ylabel("Opinions")
axs[3].set_title("Opinion Dynamics (p=1 for all users)")
axs[3].grid(True)
axs[3].set_ylim(-1, 1.05)
axs[3].set_xlim(0, simulation_steps)


plt.tight_layout()
plt.show()
"""