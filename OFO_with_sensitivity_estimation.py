import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from helper.opinion_dynamics import DGModel
from helper.clicking_functions import clicking_function_combined, minimize_combined


# Parameters
#np.random.seed(41)
np.set_printoptions(precision=4, suppress=True)
simulation_steps = 5000
num_of_triggers = simulation_steps//100
d = 5
kappa = 2 # L1 constraint on P
const_A = True
flip_A = False

# Fixed matrix and parameters
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
x_0 = np.random.uniform(-1, 1, d)
p_0 = np.random.uniform(-1, 1, d)
p_0 = (p_0 / np.linalg.norm(p_0, 1)) * kappa
gamma_p = np.random.uniform(0.01, 0.5, d)
sim = DGModel(N=d, gamma=gamma_p, A=A, x_0=x_0)
sim_ideal = DGModel(N=d, gamma=gamma_p, A=A, x_0=x_0)
theta = np.random.uniform(0, 1, d)

print("True theta:", theta)
print("True sensitivity:\n", sim.get_sensitivity())
#print("Estimated col sums: \n", estimated_sensitivity.reshape((d, d), order='C').sum(axis=0))

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
theta_error = np.zeros(num_of_triggers)

estimated_theta = 0
estimated_sensitivity = 0

trigger = 0
start_range = 0
for i in range(simulation_steps):
    if i % 100 <= 20:  # First 20 steps: give time to converge
        X[i + 1] = sim.update(p=P[trigger])
        X_ideal[i + 1] = sim_ideal.update(p=P_ideal[trigger])
    else: # Last 60 steps: measure CTR
        CTR_obs[trigger] += np.random.rand() < clicking_function_combined(P[trigger], X[i], theta=theta)
        CTR_obs_ideal[trigger] += np.random.rand() < clicking_function_combined(P_ideal[trigger], X_ideal[i], theta=theta)
        X[i + 1] = sim.update(p=P[trigger])
        X_ideal[i + 1] = sim_ideal.update(p=P_ideal[trigger])


        if i % 100 == 99: # End of trigger period: average CTR, estimate sensitivity and choose new P
            #print("Trigger", trigger + 1)
            CTR_obs[trigger] /= 80
            CTR_obs_ideal[trigger] /= 80
            #print("Observed CTR:", CTR_obs[trigger])
            #print("Ideal CTR:", clicking_function_combined(P[trigger], X[i], theta=theta))

            if not const_A:
                start_range = max(0, trigger - 50)

            result = least_squares(minimize_combined, np.concatenate((np.zeros(d*d), np.ones(d))), bounds=(np.zeros(d*d + d), (np.ones(d*d + d))), verbose=0, args=(d, P[start_range:trigger+1], CTR_obs[start_range:trigger+1])).x
            estimated_sensitivity = result[:d*d]
            estimated_sensitivity[np.abs(estimated_sensitivity) < 1e-10] = 0
            estimated_sensitivity = estimated_sensitivity.reshape((d, d), order='C')
            estimated_theta = result[d*d:]

            sensitivity_error[trigger] = np.linalg.norm(estimated_sensitivity - sim.get_sensitivity(), ord='fro')
            theta_error[trigger] = np.linalg.norm(estimated_theta - theta, ord=2)

            #print("Estimated theta:", estimated_theta)
            #print("Estimated sensitivity:\n", estimated_sensitivity)
            #print("Estimated col sums: \n", estimated_sensitivity.reshape((d, d), order='C').sum(axis=0))

            P[trigger + 1] = sim.ofo_sensitivity(prev_p=P[trigger], sensitivity=estimated_sensitivity, constraint=kappa)
            P_ideal[trigger + 1] = sim_ideal.ofo(prev_p=P_ideal[trigger], constraint=kappa)
            trigger += 1
    
    if not const_A:
        sim.evolve_A()

    if i == simulation_steps//2 and flip_A:
        sim.A[[1,4]] = sim.A[[4,1]]

    

print("Estimated theta:", estimated_theta)
print("Estimated sensitivity:\n", estimated_sensitivity)


print(P[-1])
print(P_ideal[-1])

plt.plot(sensitivity_error, label='Sensitivity error')
plt.plot(theta_error, label='Theta error')
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