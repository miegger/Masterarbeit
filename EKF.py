import numpy as np
from scipy.optimize import least_squares, root
from helper.setup_project import generate_model, clicking_function_squared
import matplotlib.pyplot as plt
from scipy.linalg import block_diag



def calculate_H(p, state):
    H = np.zeros((p.size * 2, state.size))

    sensitivity = state[0:p.size**2].reshape((p.size, p.size), order='C')
    x = sensitivity @ p
    theta = state[p.size**2:]

    for i in range(p.size):
        temp = theta[i]*np.exp(-4 * (x[i] - p[i])**2)*(-8 * (x[i] - p[i]))
        H[i, i*p.size:(i+1)*p.size] = temp * p
        H[i, p.size**2 + i] = -0.5 + np.exp(-4 * (x[i] - p[i])**2)

        H[i + p.size, i*p.size:(i+1)*p.size] = np.ones(p.size) # Row stochasticity constraint


    return H


def same_max_index(arr1, arr2):
    return np.argmax(arr1) == np.argmax(arr2)

np.set_printoptions(precision=4, suppress=True)
d = 5
num_measurements = 3000


for i in range(1):
    sim, P, CTR, true_theta, true_sensitivity = generate_model(num_measurements=num_measurements, ideal=False, clicking_function='combined')

    estimated_sensitivity = 0
    state = 0.5*np.ones(d**2 + d) # Flattened sensitivity and theta
    
    sigma_r = 10 # Std of measurement noise --> low values we trust fully the CTR
    sigma_q = 0.1 # Std of process noise

    H = np.zeros((d, d**2 + d))
    sigma = np.eye(d**2 + d)
    R = block_diag(sigma_r**2 * np.eye(d), 0 * np.eye(d))
    Q = sigma_q**2 * np.eye(d**2 + d)

    error = np.zeros(d)

    Error_sensitivity = np.zeros(num_measurements)
    Error_theta = np.zeros(num_measurements)

    #s = np.concatenate((true_sensitivity.flatten(order='C'), true_theta))
    #print(calculate_H(P[0], s) @ s, np.concatenate((CTR[i] - 0.5, np.ones(d))))

    """
    # Calculate H Rank
    Hs = []
    for i in range(num_measurements):
        Hs.append(calculate_H(P[i], state))

    H_concat = np.concatenate(Hs, axis=0)
    print("Shape of concatenated H:", H_concat.shape)
    print("Rank of concatenated H:", np.linalg.matrix_rank(H_concat))
    """


    for i in range(num_measurements):
        H = calculate_H(P[i], state)
        K = sigma @ np.transpose(H) @ np.linalg.inv(R + H @ sigma @ np.transpose(H))
        state = state + K @ (np.concatenate((CTR[i] - 0.5, np.ones(d))) - H @ state)
        #print("Estimated theta:", theta)

        sigma = sigma + Q - K @ H @ sigma

        
        #Error_sensitivity[i] = np.linalg.norm(state[0:d**2].reshape((d, d), order='C') - true_sensitivity, ord='fro')
        Error_sensitivity[i] = 100 * np.linalg.norm(np.diag(state[0:d**2].reshape((d, d), order='C') - true_sensitivity), ord=1) / np.linalg.norm(np.diag(true_sensitivity), ord=1)
        Error_theta[i] = np.linalg.norm(state[d**2:] - true_theta)

        if(i >= num_measurements - 2):
            print("Estimated sensitivity:\n", state[0:d**2].reshape((d, d), order='C'))
            print("Estimated theta:", state[d**2:])

        

    print("True sensitivity:\n", true_sensitivity)
    print("True theta:", true_theta)
    #print("True col sums: \n", true_sensitivity.sum(axis=0))

    plt.plot(Error_sensitivity)
    plt.plot(Error_theta)

    plt.show()
        

    """
    found_influencer = same_max_index(true_sensitivity.sum(axis=0), estimated_sensitivity.sum(axis=0))

    if found_influencer == False:
        print("True sensitivity: \n", true_sensitivity)
        print("True col sums: \n", true_sensitivity.sum(axis=0))

        print("Estimated sensitivity: \n", estimated_sensitivity)
        print("Estimated col sums: \n", estimated_sensitivity.sum(axis=0))
    """