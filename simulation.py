import matplotlib.pyplot as plt
import numpy as np
from helper.opinion_dynamics import FJModel, generate_adjacency_matrix



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
simulation_steps = 50
N = 15
runs = 20

# Fixed matrix and parameters
A = generate_adjacency_matrix(N)
gamma_p = np.random.uniform(0.01, 0.5, N)
x_0 = np.random.uniform(-1, 1, N)
gamma_d = 0.5 * np.abs(x_0)

# To store CTRs from all runs
all_ctrs_ofo = []
all_ctrs_star = []

position_diff_norms = np.zeros((runs, simulation_steps + 1))


for run in range(runs):
    sim = FJModel(N=N, gamma_p=gamma_p, gamma_d=gamma_d, A=A, x_0=x_0)
    sim_star = FJModel(N=N, gamma_p=gamma_p, gamma_d=gamma_d, A=A, x_0=x_0)

    p_star = sim.feedforward()

    opinions = np.zeros((simulation_steps + 1, N))
    opinions[0] = x_0
    opinions_star = np.zeros((simulation_steps + 1, N))
    opinions_star[0] = x_0

    position = np.zeros((simulation_steps + 1, N))
    number_of_clicks_ofo = np.zeros(N)
    number_of_clicks_star = np.zeros(N)


    for i in range(simulation_steps):
        if i >= 30:
            number_of_clicks_ofo = update_number_of_clicks_2(number_of_clicks_ofo, opinions[i], position[i])
            number_of_clicks_star = update_number_of_clicks_2(number_of_clicks_star, opinions_star[i], p_star)

        opinions[i + 1] = sim.update(p=position[i])
        position[i + 1] = sim.ofo(prev_p=position[i])
        opinions_star[i + 1] = sim_star.update(p=p_star)


    ctr_ofo = number_of_clicks_ofo / 20
    ctr_star = number_of_clicks_star / 20

    all_ctrs_ofo.append(ctr_ofo)
    all_ctrs_star.append(ctr_star)


    # Record the norm difference between position and opinion at each step
    position_diff_norms[run] = np.linalg.norm(position - p_star, axis=1)    

    if run == 0:
        print("P_star (first run):", p_star)
        print("Final Position (first run):", position[-1])
        print("Final Opinion (first run):", opinions[-1])

# Convert CTRs to NumPy array
overall_avg_ctr = [np.mean(np.array(all_ctrs_ofo)), np.mean(np.array(all_ctrs_star))]

print("Overall average CTR across all users and runs:", overall_avg_ctr)

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Opinion Dynamics (OFO)
axs[0].plot(opinions)
axs[0].set_ylabel("Opinions")
axs[0].set_title("Opinion Dynamics (OFO)")
axs[0].grid(True)
axs[0].set_ylim(-1, 1)
axs[0].set_xlim(0, simulation_steps)
axs[0].text(0.98, 0.9, f"CTR: {overall_avg_ctr[1]:.4f}", transform=axs[0].transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Plot 2: Opinion Dynamics (Feedforward)
axs[1].plot(opinions_star)
axs[1].set_xlabel("Time Step")
axs[1].set_ylabel("Opinions")
axs[1].set_title("Opinion Dynamics (Feedforward)")
axs[1].grid(True)
axs[1].set_ylim(-1, 1)
axs[1].set_xlim(0, simulation_steps)
axs[1].text(0.98, 0.9, f"CTR: {overall_avg_ctr[0]:.4f}", transform=axs[1].transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

# Plot all runs of ||p(k) - p*|| over time
plt.figure(3)
for run in range(runs):
    plt.plot(position_diff_norms[run], alpha=0.4)

plt.xlabel("Time Step")
plt.ylabel("||p(k) - p*||")
plt.title("OFO p(k) vs p*")
plt.grid()
plt.show()
