import matplotlib.pyplot as plt
import numpy as np
from helper.opinion_dynamics import FJModel


# Parameters
simulation_steps = 50
N = 2

# Fixed matrix and parameters
A_sub = np.array([[0.1,0.1],[0.1,0.1]])
A_stoch = np.array([[0.5,0.5],[0.5,0.5]])

gamma_p = np.random.uniform(0.01, 0.5, N)
x_0 = np.array([1,1])
gamma_d = np.random.uniform(0.01, 0.5, N)


sim = FJModel(N=N, gamma_p=gamma_p, gamma_d=gamma_d, A=A_sub, x_0=x_0)
sim_stochastic = FJModel(N=N, gamma_p=gamma_p, gamma_d=gamma_d, A=A_stoch, x_0=x_0)


opinions = np.zeros((simulation_steps + 1, N))
opinions[0] = x_0
opinions_stoch = np.zeros((simulation_steps + 1, N))
opinions_stoch[0] = x_0

position = np.array([1,1])


for i in range(simulation_steps):
    opinions[i + 1] = sim.update(p=position)
    opinions_stoch[i + 1] = sim_stochastic.update(p=position)



fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Opinion Dynamics (OFO)
axs[0].plot(opinions)
axs[0].set_ylabel("Opinions")
axs[0].set_title("Opinion Dynamics with A substochastic")
axs[0].grid(True)
axs[0].set_ylim(-1, 1.05)
axs[0].set_xlim(0, simulation_steps)

# Plot 1: Opinion Dynamics (OFO)
axs[1].plot(opinions_stoch)
axs[1].set_ylabel("Opinions")
axs[1].set_title("Opinion Dynamics with A stochastic")
axs[1].grid(True)
axs[1].set_ylim(-1, 1.05)
axs[1].set_xlim(0, simulation_steps)


plt.show()
