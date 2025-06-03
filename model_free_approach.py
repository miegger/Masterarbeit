import matplotlib.pyplot as plt
import numpy as np
from helper.opinion_dynamics import DGModel
from helper.build_delta_cr import build_delta_cr
import cvxpy as cp

def clicking_function(p, x):
    N = len(p)
    return 0.5*np.ones(N) + 0.5*p*x


def calculate_p(sim, simulation_steps, opinions, position, T):
    """

    """
    eta = 0.2
    delta = 0.001

    N = opinions.shape[1]
    number_of_clicks = np.zeros(N)
    w = 0 * np.ones(N)
    prev_norm = np.linalg.norm(opinions[0] - np.ones(N), 2)**2


    for i in range(1, simulation_steps + 1):
        opinions[i] = sim.update(p=position[i - 1])
        
        if(i % T == 0):
            v = np.random.random(N)
            print(v)
            v /= np.linalg.norm(v, ord=2)

            gradient = (N*v) / delta * (np.linalg.norm(opinions[i] - np.ones(N), 2)**2 - prev_norm)
            prev_norm = np.linalg.norm(opinions[i] - np.ones(N), 2)**2  

            w = (1 - eta) * w + eta * np.where(gradient > 0, -1, np.where(gradient < 0, 1, 0))
            #print(w)
            position[i] = w + delta * v
        
        else:
            position[i] = position[i - 1]
    

# Parameters
simulation_steps = 500
N = 5
T = 30
trials = 1

# Fixed matrix and parameters
A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])


for t in range(trials):
    x_0 = np.random.uniform(-1, 1, N)
    gamma_p = np.random.uniform(0.01, 0.5, N)
    sim = DGModel(N=N, gamma=gamma_p, A=A, x_0=x_0)
        
    opinions = np.zeros((simulation_steps + 1, N))
    opinions[0] = x_0
    position = np.zeros((simulation_steps + 1, N))

    calculate_p(sim=sim, simulation_steps=simulation_steps, opinions=opinions, position=position, T=T)

    plt.figure(0)
    plt.plot(position)
    plt.title('Position')

    plt.figure(1)
    plt.plot(opinions)
    plt.title('Opinions')

plt.show()