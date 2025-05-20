import numpy as np
import networkx as nx
from helper.opinion_dynamics import DGModel
import matplotlib.pyplot as plt


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
A = np.array([[1, 0, 0, 0, 0],[0.1, 0.9, 0, 0, 0],[0.1, 0, 0.9, 0, 0],[0, 0.8, 0, 0.2, 0],[0, 0, 0.8, 0, 0.2]])
print("Matrix power", np.linalg.matrix_power(A, 100))
#print("Eigenvalues", np.linalg.eig(A.T))

DC_out = 1/5 * np.count_nonzero(A, axis=1)
DC_in = 1/5 * np.count_nonzero(A, axis=0)
DC_in_weighted = np.sum(A, axis=0)

DG = nx.DiGraph(A)

"""
print("Out-Degree centrality", nx.out_degree_centrality(DG))
print("In-Degree centrality", nx.in_degree_centrality(DG))

print("Eigenvector_centrality", nx.eigenvector_centrality(DG))
print("Katz_centrality", nx.katz_centrality(DG))
print("Closeness_centrality", nx.closeness_centrality(DG))
"""

#exteneded de groot
gamma1 = np.diag([0.8,0.8,0.8,0.8,0.8])
gamma2 = np.diag([0,0.82,0,0,0])
gamma3 = np.diag([0,0,0,0,0.41])

x_0 = np.random.uniform(-1, 1, 5)

S = np.linalg.inv((np.eye(5) - (np.eye(5) - gamma1) @ A)) @ gamma1
print("Gamma: \n", gamma1)
print("Sensitivity: \n", S)
print("Col sum", np.sum(S, axis=0))

simulation_steps = 10
N = 5

opinions1 = np.zeros((simulation_steps + 1, N))
opinions1[0] = x_0
opinions2 = np.zeros((simulation_steps + 1, N))
opinions2[0] = x_0
opinions3 = np.zeros((simulation_steps + 1, N))
opinions3[0] = x_0

position1 = np.array([1,1,1,1,1])
position2 = np.array([0,1,0,0,0])
position3 = np.array([0,0,0,0,1])

sim1 = DGModel(N=N, gamma=gamma1, A=A, x_0=x_0)
sim2 = DGModel(N=N, gamma=gamma2, A=A, x_0=x_0)
sim3 = DGModel(N=N, gamma=gamma3, A=A, x_0=x_0)

for i in range(simulation_steps):
    opinions1[i + 1] = sim1.update(p=position1)
    opinions2[i + 1] = sim2.update(p=position2)
    opinions3[i + 1] = sim3.update(p=position3)


"""
ig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

axs[0].plot(opinions1)
axs[0].set_ylabel("Opinions")
axs[0].set_title("Opinion Dynamics, recommendation for all users")
axs[0].grid(True)
axs[0].set_ylim(-1, 1)
axs[0].legend(["user 1", "user 2", "user 3", "user 4", "user 5"])
axs[0].set_xlim(0, simulation_steps)

axs[1].plot(opinions2)
axs[1].set_ylabel("Opinions")
axs[1].set_title("Opinion Dynamics, recommendation for user 2")
axs[1].grid(True)
axs[1].set_ylim(-1, 1)
axs[1].set_xlim(0, simulation_steps)

axs[2].plot(opinions3)
axs[2].set_xlabel("Time Step")
axs[2].set_ylabel("Opinions")
axs[2].set_title("Opinion Dynamics, recommendation for user 5")
axs[2].grid(True)
axs[2].set_ylim(-1, 1)
axs[2].set_xlim(0, simulation_steps)

plt.show()
"""