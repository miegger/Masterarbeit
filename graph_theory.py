import numpy as np
import networkx as nx
from helper.opinion_dynamics import DGModel, Delta_DGModel
import matplotlib.pyplot as plt
import scipy.linalg as sla

def clicking_function_combined(p, opinions, theta=0.5*np.ones(5)):
  """Clicking function model: 0.5(1-theta) + theta * exp(-4 * (opinions - p)**2)"""
  return 0.5*(np.ones_like(p) - theta) + theta*np.exp(-4*(opinions - p)**2)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
#A = np.array([[1, 0, 0, 0, 0],[0.1, 0.9, 0, 0, 0],[0.1, 0, 0.9, 0, 0],[0, 0.8, 0, 0.2, 0],[0, 0, 0.8, 0, 0.2]])
#print("Matrix power", np.linalg.matrix_power(A, 100))
#print("Eigenvalues", np.linalg.eig(A.T))


#exteneded de groot
gamma = np.diag(np.random.uniform(0.01, 0.5, 5))
delta = np.diag([1,0,0,0,0])

x_0 = np.random.uniform(-1, 1, 5)
print(0.4 * x_0[1] + 0.6 * x_0[4])
print(x_0)

S = np.linalg.inv((np.eye(5) - (np.eye(5) - gamma) @ A)) @ gamma
print("Sensitivity: \n", S @ np.diag(delta))
print("Col sum: ", np.sum(S, axis=0))

S_delta = np.linalg.inv((np.eye(5) - (np.eye(5) - delta @ gamma) @ A)) @ delta @ gamma
print("SS???", S_delta) #NOOOO

def build_Phi(A, Gamma, Delta):
    I = np.eye(A.shape[0])
    return (I - Delta.dot(Gamma)).dot(A)

def analytic_steady_state(A, Gamma, Delta, p):
    Phi = build_Phi(A, Gamma, Delta)
    b = Delta.dot(Gamma).dot(p)
    I = np.eye(A.shape[0])
    M = I - Phi
    # try direct solve; if singular, return pseudo-inverse solution
    try:
        x_star = sla.solve(M, b)
        method = "direct_solve"
    except Exception as e:
        x_star = np.linalg.pinv(M).dot(b)
        method = "pinv"
    return x_star, Phi, b, method

print(analytic_steady_state(A, gamma, delta, np.ones(5)))


simulation_steps = 100
N = 5

opinions = np.zeros((simulation_steps + 1, N))
opinions[0] = x_0


sim = Delta_DGModel(N=N, gamma=np.diag(gamma), A=A, x_0=x_0, delta = np.diag(delta))
sim_test = DGModel(N=N, gamma=np.diag(gamma), A=A, x_0=x_0)

for i in range(simulation_steps):
    opinions[i + 1] = sim.update()

print(opinions[-1])



plt.plot(opinions)
plt.show()
