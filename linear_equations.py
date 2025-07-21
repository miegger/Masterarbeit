import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linprog
from helper.setup_project import generate_model

def generate_measurement_matrix_C(G):
    d = G.shape[0] # number of users
    b = np.ones(d)

    matrix = np.kron(np.eye(d), np.ones(d))

    second_part = []

    for i in range(5):
        temp = np.eye(d)
        for j in range(d):
            if i != j:
                if G[i, j] != 0:
                    temp[j, j] = -1*G[i, i] / G[i, j]
                    temp[j] = temp[j] + temp[i]
                    b = np.append(b,1)
                else:
                    temp[j, j] = 1
                    b = np.append(b, 0)
        #temp = temp + temp[i]
        second_part.append(np.delete(temp, i, axis=0))
    #return block_diag(*second_part), b
    return np.concatenate((matrix, block_diag(*second_part)), axis=0), b

def find_max_z(N, x_est, c):
    # Construct bounds: 0 <= x_est + N @ z <= 1
    # This is equivalent to:
    #     -N @ z <= x_est
    #      N @ z <= 1 - x_est

    # We write this in the form: A_ub @ z <= b_ub
    A_ub = np.vstack([-N, N])
    b_ub = np.hstack([x_est, 1 - x_est])

    # Objective: maximize c @ N @ z <=> maximize (c @ N) @ z
    c_transformed = c @ N
    objective = -c_transformed  # negate for minimization

    # Solve
    result = linprog(objective, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * 5, method='highs')


    if result.success:
        z_opt = result.x
        x = x_est + N @ z_opt
        print("Optimal z:", z_opt)
        #print("x within bounds:", x.reshape((5,5), order='C'))
        print("Objective value (max c @ x):", c @ x)
    else:
        print("Optimization failed:", result.message)




np.set_printoptions(precision=4)
sim, P, CTR, G, true_sensitivity = generate_model(num_measurements=2, clicking_function='squared')
d = G.shape[1]
#print(G[0])

A_C, b = generate_measurement_matrix_C(G[0])
#A_C_dash, b_dash = generate_measurement_matrix_C(G[1])
#print(np.linalg.matrix_rank(np.vstack([A_C, A_C_dash])))
#print(np.vstack([A_C, A_C_dash]).shape)

estimated_sensitivity, res, rank, sing_values = np.linalg.lstsq(A_C, b)

estimated_sensitivity[np.abs(estimated_sensitivity) < 1e-10] = 0
#estimated_sensitivity = estimated_sensitivity.reshape((d, d), order='C')

U, S, Vt = np.linalg.svd(A_C)

null_space = Vt.T[:, S < 1e-10]
null_space[np.abs(null_space) < 1e-10] = 0


c1 = np.array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0])
c2 = np.array([0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0])
c3 = np.array([0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0])
c4 = np.array([0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0])
c5 = np.array([0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])
max_z = find_max_z(null_space, estimated_sensitivity, c1)
max_z = find_max_z(null_space, estimated_sensitivity, c2)
max_z = find_max_z(null_space, estimated_sensitivity, c3)
max_z = find_max_z(null_space, estimated_sensitivity, c4)
max_z = find_max_z(null_space, estimated_sensitivity, c5)





print("Rank:", rank)

print("True sensitivity: \n", true_sensitivity)
print("Estimated col sums: \n", true_sensitivity.sum(axis=0))

print("Estimated sensitivity: \n", estimated_sensitivity)
#print("Estimated col sums: \n", estimated_sensitivity.sum(axis=0))

#print(A_C @ true_sensitivity.flatten('C'))
#print(b)


