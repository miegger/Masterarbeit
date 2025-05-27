import numpy as np

def build_delta_cr(p_1, p_2):
    N = len(p_1)
    delta_cr = np.zeros((N, N**2))

    for i in range(N):
        for j in range(N):
            delta_cr[i,j*N + i] = 0.5 * (p_1[i] * p_1[j] - p_2[i] * p_2[j])

    return delta_cr