import matplotlib.pyplot as plt
import numpy as np
from helper.opinion_dynamics import DGModel


# Parameters
simulation_steps = 30
N = 5
trials = 50

# Fixed matrix and parameters
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])

# To store positions from all runs
all_positions = np.zeros((trials, N))
all_norms = np.zeros((trials))
all_norms_comparision = np.zeros((trials))


for t in range(trials):
    x_0 = np.random.uniform(-1, 1, N)
    gamma_p = np.random.uniform(0.01, 0.5, N)

    sim = DGModel(N=N, gamma=gamma_p, A=A, x_0=x_0)
    sim_comparison = DGModel(N=N, gamma=gamma_p, A=A, x_0=x_0)

    opinions = np.zeros((simulation_steps + 1, N))
    opinions[0] = x_0
    opinions_comparison = np.zeros((simulation_steps + 1, N))
    opinions_comparison[0] = x_0

    position = np.zeros((simulation_steps + 1, N))
    delta = np.zeros((simulation_steps + 1, N))
    delta[0] = np.ones(N)

    for i in range(simulation_steps):
        opinions[i + 1] = sim.update(p=position[i])
        position[i + 1] = sim.ofo(prev_p=position[i], constraint=2)

        opinions_comparison[i + 1] = sim_comparison.update(p=np.array([0, 1, 0, 0, 1]))

    all_positions[t] = position[-1]
    all_norms[t] = 0.5 * np.linalg.norm(opinions[-1] - np.ones(N), ord=2)**2
    all_norms_comparision[t] = 0.5 * np.linalg.norm(opinions_comparison[-1] - np.ones(N), ord=2)**2

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