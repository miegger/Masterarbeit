import matplotlib.pyplot as plt
import numpy as np
from helper.opinion_dynamics import DGModel
import cvxpy as cp

# Parameters
simulation_steps = 50
N = 5
trials = 10
T = 50
kappa = 2

A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])

all_positions = np.zeros((trials, N))
all_positions_ofo = np.zeros((trials, N))
all_positions_ofo_proj = np.zeros((trials, N))


for t in range(trials):
    gamma_p = np.random.uniform(0.01, 0.99, N)
    x_0 = np.random.uniform(-1, 1, N)

    sim1 = DGModel(N, gamma_p, A, x_0)
    sim2 = DGModel(N, gamma_p, A, x_0)
    gamma_p = np.diag(gamma_p)

    # Solve problem a priori
    p = cp.Variable(N, integer=True)

    A_tilde = (np.eye(N) - gamma_p) @ A
    x = np.linalg.inv(np.eye(N) - A_tilde) @ gamma_p @ p

    objective = cp.Minimize(cp.norm(x - np.ones(N), 1))
    constraints = [p >= -1, p <= 1, cp.norm(p, 1) <= kappa]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    all_positions[t] = p.value

    #print(all_positions[t])

    # Solve problem with OFO
    position = np.zeros((simulation_steps + 1, N))
    opinions = np.zeros((simulation_steps + 1, N))
    opinions[0] = x_0
    p_not_int = np.zeros(N)
    
    for i in range(simulation_steps):
        opinions[i + 1] = sim1.update(position[i])
        p_not_int = sim1.ofo(prev_p=p_not_int, constraint=kappa)

        if kappa is None:
            position[i + 1] = np.round(p_not_int)
        else:
            binary_p = np.zeros(N)
            top_indices = np.argpartition(p_not_int, -kappa)[-kappa:]
            binary_p[top_indices] = 1
            position[i + 1] = binary_p

    all_positions_ofo[t] = position[-1]

    position = np.zeros((simulation_steps + 1, N))
    opinions = np.zeros((simulation_steps + 1, N))
    opinions[0] = x_0
    p_not_int = np.zeros(N)


    for i in range(simulation_steps):
        opinions[i + 1] = sim2.update(position[i])
        p_not_int = sim2.ofo(prev_p=p_not_int, constraint=kappa)

        p2 = cp.Variable(N, integer=True)
        objective = cp.Minimize(cp.norm1(p2 - p_not_int))
        constraints = [p2 >= -1, p2 <= 1, cp.norm(p, 1) <= kappa]
        problem2 = cp.Problem(objective, constraints)
        problem2.solve()
        
        position[i + 1] = p2.value

    all_positions_ofo_proj[t] = position[-1]



# Create the boxplot
plt.figure(figsize=(8, 5))
plt.plot(np.mean(all_positions, axis = 0), '-o')
plt.plot(np.mean(all_positions_ofo, axis = 0), '-o')
plt.plot(np.mean(all_positions_ofo_proj, axis = 0), '-o')
plt.legend(['MILP solver', 'OFO with rounding', 'OFO with projection'])

# Labeling
plt.ylabel("Position")
plt.title(f"Mean positions from 50 trials, constraint on ||p||_1 <= {kappa} and p integer")
plt.xticks(range(0, N), [f'User {i}' for i in range(1, N + 1)])
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(-0.1,1)
plt.tight_layout()
plt.show()