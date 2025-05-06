import numpy as np
#import networkx as nx
from helper.opinion_dynamics import DGModel
import matplotlib.pyplot as plt


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
A = np.array([[0.45, 0.05, 0.1, 0.3, 0.1],[0, 0.5, 0, 0, 0.5],[0.3, 0.05, 0.35, 0.15, 0.15],[0, 0.15, 0.3, 0.55, 0],[0.5, 0.2, 0, 0, 0.3]])


print(A)

print("Col sum", np.sum(A, axis=0))
print("Row sum", np.sum(A, axis=1))

print("Matrix power", np.linalg.matrix_power(A, 100))


# Below simulations with a constant gamma

ga = np.linspace(0.1, 0.9, 10)
col_sum = np.zeros([10,5])
eigenvector_centrality = np.zeros([10,5])


for i, g in enumerate(ga):
    gamma = np.diag([g]*5)
    S = np.linalg.inv(np.eye(5) - ((np.eye(5) - gamma) @ A)) @ gamma
    col_sum[i] = np.sum(S, axis=0)
    eigenvector_centrality[i] = np.abs(np.linalg.eig(S.T).eigenvectors[:,0])
    print(eigenvector_centrality[i])

for i in range(col_sum.shape[0]):
    plt.plot(eigenvector_centrality[i], 'o-', label=f'$\\gamma = {ga[i]:.1f}$')
 

plt.xticks(ticks=[0, 1, 2, 3, 4], labels=[1, 2, 3, 4, 5])
plt.xlabel('Column')
plt.ylabel('Column Sum')
plt.title('Column sum of the sensitivity matrix')
plt.legend(ncol=3)
plt.grid()
plt.show()






"""
# Below simulation with different gammas
runs = 10
eigenvector_centrality = np.zeros([runs,5])
col_sum = np.zeros([runs,5])
row_sum = np.zeros([runs,5])
gammas = np.zeros([runs,5])


for i in range(runs):
    ga = np.linspace(0, 1, 50)
    gamma = np.zeros([5,5])



    if(i == 0):
        gamma = np.diag([0.99,0.01,0.99,0.99,0.01])
    elif(i == 1):
        gamma = np.diag([0.9,0,0.8,0.7,0])
    elif(i == 2):
        gamma = np.diag([0.01,0.01,0.01,0.99,0.01])
    elif(i == 3):
        gamma = np.diag([0.2,0,0.5,0.8,0.1])
    elif(i == 4):
        gamma = np.diag([0.99,0,0.99,0.99,0.01])
    elif(i == 5):
        gamma = np.diag([0.99,0.01,0.99,0.99,0])
    elif(i == 6):
        gamma = np.diag([0.5,0,0.5,0.5,0])
    else:
        for j in range(5):
            gamma[j][j] = ga[np.random.randint(50)]

    for j in range(5):
        gamma[j][j] = ga[np.random.randint(50)]


    S = np.linalg.inv(np.eye(5) - ((np.eye(5) - gamma) @ A)) @ gamma
    eigenvalues, eigenvectors = np.linalg.eig(S.T)
    max_index = np.argmax(np.abs(eigenvalues))
    largest_eigenvalue = eigenvalues[max_index]
    eigenvector_centrality[i] = np.abs(eigenvectors[:, max_index])


    col_sum[i] = np.sum(S, axis=0)
    row_sum[i] = np.sum(S, axis=1)
    gammas[i] = np.diagonal(gamma)

    print(eigenvector_centrality[i], gammas[i])


for i in range(col_sum.shape[0]):
    formatted = ", ".join([f"{val:.2f}" for val in gammas[i]])
    plt.plot(eigenvector_centrality[i], 'o:', label=f'$\\gamma = [{formatted}]$')

    # Find max value and its index
    max_idx = np.argmax(eigenvector_centrality[i])
    max_val = eigenvector_centrality[i][max_idx]

    # Mark the max point
    plt.scatter(max_idx, max_val, s=150, marker='x')

plt.xticks(ticks=[0, 1, 2, 3, 4], labels=[1, 2, 3, 4, 5])
plt.xlabel('Column')
plt.ylim([0,2])
plt.ylabel('Column Sum')
plt.title('Eigenvector centrality of the sensitivity matrix')
plt.legend(ncol=3)
plt.grid()
plt.show()


for i in range(col_sum.shape[0]):
    formatted = ", ".join([f"{val:.2f}" for val in gammas[i]])
    plt.plot(col_sum[i], 'o:', label=f'$\\gamma = [{formatted}]$')

    # Find max value and its index
    max_idx = np.argmax(col_sum[i])
    max_val = col_sum[i][max_idx]

    # Mark the max point
    plt.scatter(max_idx, max_val, s=150, marker='x')

plt.xticks(ticks=[0, 1, 2, 3, 4], labels=[1, 2, 3, 4, 5])
plt.xlabel('Column')
plt.ylim([0,4.5])
plt.ylabel('Column Sum')
plt.title('Column sum of the sensitivity matrix')
plt.legend(ncol=3)
plt.grid()
plt.show()
"""

"""
#exteneded de groot
gamma1 = 0.5 * np.diag([1,1,1,1,1])
gamma2 = 0.5 * np.diag([0,1,0,0,1])
gamma3 = 0.5 * np.diag([0,0,0,0,1])

S = np.linalg.inv(np.eye(5) - (np.eye(5) - gamma1) @ A) @ gamma1
print("Sensitivity 1: \n", S)
print("Col sum", np.sum(S, axis=0))
print("Row sum", np.sum(S, axis=1))

S = np.linalg.inv(np.eye(5) - (np.eye(5) - gamma2) @ A) @ gamma2
print("Sensitivity 2: \n", S)
print("Col sum", np.sum(S, axis=0))
print("Row sum", np.sum(S, axis=1))

S = np.linalg.inv(np.eye(5) - (np.eye(5) - gamma3) @ A) @ gamma3
print("Sensitivity 3: \n", S)
print("Col sum", np.sum(S, axis=0))
print("Row sum", np.sum(S, axis=1))
"""




"""
x_0 = np.random.uniform(-1, 1, 5)
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
position2 = np.array([0,1,0,0,1])
position3 = np.array([0,0,0,1,0])

sim1 = DGModel(N=N, gamma=gamma1, A=A, x_0=x_0)
sim2 = DGModel(N=N, gamma=gamma2, A=A, x_0=x_0)
sim3 = DGModel(N=N, gamma=gamma3, A=A, x_0=x_0)

for i in range(simulation_steps):
    opinions1[i + 1] = sim1.update(p=position1)
    opinions2[i + 1] = sim2.update(p=position2)
    opinions3[i + 1] = sim3.update(p=position3)


ig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

axs[0].plot(opinions1)
axs[0].set_ylabel("Opinions")
axs[0].set_title("Opinion Dynamics all users")
axs[0].grid(True)
axs[0].set_ylim(-1, 1)
axs[0].legend(["user 1", "user 2", "user 3", "user 4", "user 5"])
axs[0].set_xlim(0, simulation_steps)

axs[1].plot(opinions2)
axs[1].set_ylabel("Opinions")
axs[1].set_title("Opinion Dynamics 2 users")
axs[1].grid(True)
axs[1].set_ylim(-1, 1)
axs[1].set_xlim(0, simulation_steps)

axs[2].plot(opinions3)
axs[2].set_xlabel("Time Step")
axs[2].set_ylabel("Opinions")
axs[2].set_title("Opinion Dynamics 1 users")
axs[2].grid(True)
axs[2].set_ylim(-1, 1)
axs[2].set_xlim(0, simulation_steps)

plt.show()
"""
