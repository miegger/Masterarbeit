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

def count_ranking_errors(arr1, arr2):
    # Get sorted indices of both arrays (from highest to lowest)
    rank1 = sorted(range(len(arr1)), key=lambda i: -arr1[i])
    rank2 = sorted(range(len(arr2)), key=lambda i: -arr2[i])
    
    # Count mismatches
    errors = sum(1 for i in range(len(arr1)) if rank1[i] != rank2[i])
    return errors

def same_max_index(arr1, arr2):
    return np.argmax(arr1) == np.argmax(arr2)

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
    
    # First LP: maximize c^T N z
    res_max = linprog(-c_transformed, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * N.shape[1], method='highs')

    # Second LP: minimize c^T N z
    res_min = linprog(c_transformed, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * N.shape[1], method='highs')

    if res_max.success and res_min.success:
        max_val = -res_max.fun
        min_val = res_min.fun
        max_error = max(abs(max_val), abs(min_val))
        print("Max error |c^T N z| =", max_error)
        print("x:", (x_est + N @ res_max.x).reshape((d, d), order='C'))
        print("x:", (x_est + N @ res_min.x).reshape((d, d), order='C'))
    else:
        if not res_max.success:
            print("Max LP failed:", res_max.message)
        if not res_min.success:
            print("Min LP failed:", res_min.message)


def clicking_function_squared(p, opinions, theta = 0.25*np.ones(5)):
  """Clicking function model: 1 - theta * (opinions - p)**2"""
  return np.ones_like(p) - theta*(opinions - p)**2

def estimate_derivative(P, sensitivity, d, h=0.01):
    y = np.zeros(d)
    for i in range(d):
        p_dash = np.copy(P)
        p_dash[i]+= h
        if(p_dash[i] > 1):
            p_dash[i] -= 2*h

        X = sensitivity @ P
        x_dash = sensitivity @ p_dash

        diff = ((clicking_function_squared(p_dash, x_dash) - clicking_function_squared(P, X)) / h)
        print(diff)
        y[i] = diff[i]

    return y

def estimate_derivative2(P, sensitivity, d, h=0.1):
    y = np.zeros(d)
    for i in range(1):
        p_dash = np.copy(P)
        p_dash[i]+= h
        if(p_dash[i] > 1):
            p_dash[i] -= 2*h

        X = sensitivity @ P
        x_dash = sensitivity @ p_dash

        diff = ((clicking_function_squared(p_dash, x_dash) - clicking_function_squared(P, X)) / (X - x_dash))
        print(diff)
        #y[i] = diff[i]

    #return y

np.set_printoptions(precision=4)

num_missmatch = np.zeros(50)
found_influencer = np.zeros(50)

for i in range(50):
    sim, P, CTR, G, true_sensitivity = generate_model(num_measurements=1, ideal=True, clicking_function='squared')
    d = G.shape[1]

    
    A_C, b = generate_measurement_matrix_C(G[0])
    estimated_sensitivity, res, rank, sing_values = np.linalg.lstsq(A_C, b)
    estimated_sensitivity[np.abs(estimated_sensitivity) < 1e-10] = 0


    """
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
    """


    #print("Rank:", rank)

    estimated_sensitivity = estimated_sensitivity.reshape((d, d), order='C')


    num_missmatch[i] = count_ranking_errors(true_sensitivity.sum(axis=0), estimated_sensitivity.sum(axis=0))
    found_influencer[i] = same_max_index(true_sensitivity.sum(axis=0), estimated_sensitivity.sum(axis=0))

    if found_influencer[i] == False:
        print("True sensitivity: \n", true_sensitivity)
        print("True col sums: \n", true_sensitivity.sum(axis=0))

        print("Estimated sensitivity: \n", estimated_sensitivity)
        print("Estimated col sums: \n", estimated_sensitivity.sum(axis=0))


print("Average number of ranking errors:", np.mean(num_missmatch))
print("False nr. 1 influencer:", np.sum(found_influencer == False))