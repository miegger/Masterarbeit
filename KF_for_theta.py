import numpy as np
from scipy.optimize import least_squares, root
from helper.setup_project import generate_model, clicking_function_squared
import matplotlib.pyplot as plt



def find_sensitivity(x, theta, d, p, ctr):
    sensitivity = x.reshape((d, d), order='C')
    x = sensitivity @ p.T

    row_stochastic = np.sum(sensitivity, axis=1) - np.ones(d)
    clicking = 0.5*(1 - theta.T) + theta.T*np.exp(-4*(x.flatten(order='F') - p.flatten(order='C'))**2) - ctr.flatten(order='C')

    return np.concatenate((clicking, row_stochastic))

def calculate_H(P, X):
    return np.diag((np.exp(-4 * (X - P)**2) - 0.5))


def same_max_index(arr1, arr2):
    return np.argmax(arr1) == np.argmax(arr2)

np.set_printoptions(precision=4, suppress=True)
d = 5
num_measurements = 100


for i in range(1):
    sim, P, CTR, true_theta, true_sensitivity = generate_model(num_measurements=num_measurements, ideal=True, clicking_function='combined')

    estimated_sensitivity = 0
    theta = np.ones(d)
    
    sigma_r = 1 # Std of measurement noise --> low values we trust fully the CTR
    sigma_q = 10 # Std of process noise

    H = np.eye(d)
    sigma = np.eye(d)
    R = sigma_r**2 * np.eye(d)
    Q = sigma_q**2 * np.eye(d)


    Error_sensitivity = np.zeros(num_measurements)
    Error_theta = np.zeros(num_measurements)

    for i in range(num_measurements):
        result = least_squares(find_sensitivity, 0.5*np.ones(d*d), bounds=(np.zeros(d*d), np.ones(d*d)), verbose=0, args=(theta, d, P[i], CTR[i])).x
        estimated_sensitivity = result
        estimated_sensitivity[np.abs(estimated_sensitivity) < 1e-10] = 0

        Error_sensitivity[i] = np.linalg.norm(estimated_sensitivity.reshape((d, d), order='C') - true_sensitivity, ord='fro')
        Error_theta[i] = np.linalg.norm(theta - true_theta)
        
        H = calculate_H(P[i], estimated_sensitivity.reshape((d, d), order='C') @ P[i])
        K = sigma @ np.transpose(H) @ np.linalg.inv(R + H @ sigma @ np.transpose(H))
        theta = theta + K @ (CTR[i] - 0.5 - H @ theta)

        sigma = sigma + Q - K @ H @ sigma

        if(i >= num_measurements - 2):
            print("Estimated sensitivity:\n", estimated_sensitivity.reshape((d, d), order='C'))
            print("Estimated theta:",theta)

    print("True sensitivity:\n", true_sensitivity)
    print("True theta:", true_theta)
    #print("True col sums: \n", true_sensitivity.sum(axis=0))

    plt.plot(Error_sensitivity, label='Sensitivity Error')
    plt.plot(Error_theta, label='Theta Error')
    plt.show()


    """
    found_influencer = same_max_index(true_sensitivity.sum(axis=0), estimated_sensitivity.sum(axis=0))

    if found_influencer == False:
        print("True sensitivity: \n", true_sensitivity)
        print("True col sums: \n", true_sensitivity.sum(axis=0))

        print("Estimated sensitivity: \n", estimated_sensitivity)
        print("Estimated col sums: \n", estimated_sensitivity.sum(axis=0))
    """