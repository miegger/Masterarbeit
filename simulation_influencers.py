import matplotlib.pyplot as plt
import numpy as np
from helper.opinion_dynamics import DGModel, generate_adjacency_matrix


def calculate_clicking_probability(x, p):
    return 1 - 0.25 * (x - p)**2

def update_number_of_clicks(number_of_clicks, x, p):
    for i in range(len(number_of_clicks)):
        number_of_clicks[i] += np.random.rand() < calculate_clicking_probability(x[i], p[i])
    return number_of_clicks

def calculate_clicking_probability_2(x, p):
    return 1 - 0.5 * np.abs(x - p)

def update_number_of_clicks_2(number_of_clicks, x, p):
    for i in range(len(number_of_clicks)):
        number_of_clicks[i] += np.random.rand() < calculate_clicking_probability_2(x[i], p[i])
    return number_of_clicks

# Parameters
simulation_steps = 30
N = 5

# Fixed matrix and parameters
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
gamma_p = np.random.uniform(0.01, 0.5, N)
x_0 = np.random.uniform(-1, 1, N)

# To store CTRs from all runs
all_ctrs_ofo = []
all_ctrs_star = []

sim_all = DGModel(N=N, gamma=gamma_p, A=A, x_0=x_0)
sim = DGModel(N=N, gamma=gamma_p, A=A, x_0=x_0)

opinions = np.zeros((simulation_steps + 1, N))
opinions[0] = x_0
opinions_all = np.zeros((simulation_steps + 1, N))
opinions_all[0] = x_0

position = np.zeros((simulation_steps + 1, N))
delta = np.zeros((simulation_steps + 1, N))
delta[0] = np.ones(N)

for i in range(simulation_steps):
    opinions[i + 1] = sim.update(p=position[i])
    position[i + 1] = sim.ofo(prev_p=position[i], constraint=None)
    
print("Constraint None: position", position[-1], "cost function", 0.5 * np.linalg.norm((opinions[-1] - np.ones(N))**2, ord=2))
fig, axs = plt.subplots(3, 1, figsize=(24, 8), sharex=True)
fig2, ax2 = plt.subplots(3, 1, figsize=(24, 8), sharex=True)

   
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


"""
# Plot 2: Opinion Dynamics (Feedforward)
axs[3].plot(opinions_all)
axs[3].set_xlabel("Time Step")
axs[3].set_ylabel("Opinions")
axs[3].set_title("Opinion Dynamics (p=1 for all users)")
axs[3].grid(True)
axs[3].set_ylim(-1, 1.05)
axs[3].set_xlim(0, simulation_steps)
"""

plt.tight_layout()
plt.show()
