import numpy as np
import networkx as nx
from helper.opinion_dynamics import DGModel, Delta_DGModel, FJModel
import matplotlib.pyplot as plt
import scipy.linalg as sla
from scipy.optimize import least_squares, minimize


def clicking_function_combined(p, opinions, theta=0.5*np.ones(5)):
  """Clicking function model: 0.5(1-theta) + theta * exp(-4 * (opinions - p)**2)"""
  return 0.5*(np.ones_like(p) - theta) + theta*np.exp(-4*(opinions - p)**2)

def top_kappa_columns(M: np.ndarray, kappa: int):
    # compute column sums
    col_sums = M.sum(axis=0)
    
    # get indices of top-kappa sums
    top_indices = np.argsort(col_sums)[-kappa:]
    
    # build binary vector
    mask = np.zeros(M.shape[1], dtype=int)
    mask[top_indices] = 1
    
    return mask

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
#A = np.array([[1, 0, 0, 0, 0],[0.1, 0.9, 0, 0, 0],[0.1, 0, 0.9, 0, 0],[0, 0.8, 0, 0.2, 0],[0, 0, 0.8, 0, 0.2]])
#print("Matrix power", np.linalg.matrix_power(A, 100))
#print("Eigenvalues", np.linalg.eig(A.T))






#exteneded de groot
gamma_p = np.array([0.2,0.4,0.3,0.5,0.3])
gamma_d = np.array([0.2,0.2,0.2,0.2,0.2])

x_0 = np.random.uniform(-1,1,5)

simulation_steps = 100
N = 5

opinions = np.zeros((simulation_steps + 1, N))
opinions[0] = x_0

opinions_FJ = np.zeros((simulation_steps + 1, N))
opinions_FJ[0] = x_0

opinions_FJ_2 = np.zeros((simulation_steps + 1, N))
opinions_FJ_2[0] = x_0

p = np.array([0, 0, 0, 0, 1])
p2 = np.array([0, 1, 0, 0, 0])

d = np.random.normal(0, 0.2, N)
sim_FJ = FJModel(N=N, gamma_p=gamma_p, gamma_d=gamma_d, A=A, x_0=x_0, d=x_0)
print("FJ Sensitivity:\n", np.sum(sim_FJ.get_sensitivity(), axis=0))
feedforward = top_kappa_columns(sim_FJ.get_sensitivity(), 1)

sim_FJ_2 = FJModel(N=N, gamma_p=gamma_p, gamma_d=gamma_d, A=A, x_0=x_0, d=d)
print("FJ2 Sensitivity:\n", np.sum(sim_FJ_2.get_sensitivity(), axis=0))


sim = DGModel(N=N, gamma=gamma_p, A=A, x_0=x_0)
#print("DG Sensitivity:\n", np.sum(sim.get_sensitivity(), axis=0))
#feedforward = minimize(lambda p: -1*np.sum(sim.get_sensitivity() @ p), np.zeros(N), bounds=[(-1,1)]*d, constraints = {'type': 'eq','fun': lambda p: 1 - np.sum(np.abs(p))})


for i in range(simulation_steps):
    opinions[i + 1] = sim.update(p)
    opinions_FJ[i + 1] = sim_FJ.update(p)
    opinions_FJ_2[i + 1] = sim_FJ_2.update(p2)

print(np.sum(opinions[-1]))
print("FJ", np.sum(opinions_FJ[-1]))
print("FJ2", np.sum(opinions_FJ_2[-1]))



plt.plot(np.sum(opinions, axis=1), label="DG")
plt.plot(np.sum(opinions_FJ, axis=1), label="FJ")
plt.plot(np.sum(opinions_FJ_2, axis=1), label="FJ2")
plt.legend()
plt.show()
