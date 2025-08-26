import numpy as np
from helper.opinion_dynamics import DGModel

def clicking_function_linear(p, opinions):
  """Clicking function model: 1/2 + 1/2 opinions * p"""
  return 0.5 * np.ones_like(p) + 0.5 * opinions * p

def clicking_function_squared(p, opinions, theta = 0.25*np.ones(5)):
  """Clicking function model: 1 - theta * (opinions - p)**2"""
  return np.ones_like(p) - theta*(opinions - p)**2

def clicking_function_exponential(p, opinions, theta=np.ones(5)):
  """Clicking function model: exp(-theta * (opinions - p)**2)"""
  return np.exp(-theta*(opinions - p)**2)

def clicking_function_exponential_half(p, opinions):
  """Clicking function model: 0.5 + 0.5*exp(-4 * (opinions - p)**2)"""
  return 0.5*np.ones_like(p) + 0.5*np.exp(-4*(opinions - p)**2)

def clicking_function_combined(p, opinions, theta=0.5*np.ones(5)):
  """Clicking function model: 0.5(1-theta) + theta * exp(-4 * (opinions - p)**2)"""
  return 0.5*(np.ones_like(p) - theta) + theta*np.exp(-4*(opinions - p)**2)

def clicking_function_squared_combined(p, opinions, theta=0.5*np.ones(5)):
  """Clicking function model: 0.5(1-theta) + theta * (1 - 0.25(opinions - p)**2)"""
  return 0.5*(np.ones_like(p) - theta) + theta*(np.ones_like(p) - 0.25*(opinions - p)**2)

def kalman_filter(delta_p, delta_cr, true_sensitivity, N):
    sigma_r = 0.1 # Std of measurement noise --> low values we trust fully the CTR
    sigma_q = 0.1 # Std of process noise

    H = np.eye(N)
    l = H.flatten(order='F')
    sigma = np.eye(N**2)
    R = sigma_r**2 * np.eye(N)
    Q = sigma_q**2 * np.eye(N**2)

    error = np.zeros(len(delta_p))

    for i in range(len(delta_p)):
        kroeneker = np.kron(np.transpose(delta_p[i]), np.eye(N))
        K = sigma @ np.transpose(kroeneker) @ np.linalg.inv(R + kroeneker @ sigma @ np.transpose(kroeneker))
        l = l + K @ (delta_cr[i] - kroeneker @ l)

        sigma = sigma + Q - K @ kroeneker @ sigma
        
        #denominator = np.where(np.abs(true_sensitivity[i]) <= 1e-3, 1, np.abs(true_sensitivity[-1]))
        #error[i] = 100*np.mean(np.abs(l.reshape((N, N),order='F') - true_sensitivity[i]) / denominator)

        #error[i] = np.linalg.norm(l.reshape((N, N), order='F') - true_sensitivity[i], ord='fro')
        #print(np.round(delta_cr[i], 3), np.round(true_sensitivity[i] @ delta_p[i], 3))
        #print(np.round(delta_cr[i], 3), np.round(l.reshape((N, N), order='F') @ delta_p[i], 3), "\n")

    return l.reshape((N, N), order='F')


def generate_model(num_measurements, ideal=True, clicking_function=['squared', 'exponential', 'exponential_half', 'linear', 'combined', 'squared combined']):
  num_samples = num_measurements
  d = 5  # Dimension of p and x
  A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
  
  #A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0.1, 0.45, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
  
  x_0 = np.random.uniform(-1, 1, d)
  gamma_p = np.random.uniform(0.01, 0.5, d)
  sim = DGModel(N=d, gamma=gamma_p, A=A, x_0=x_0)
  theta = np.random.normal(0.25, 0.1, d)
  if clicking_function == 'exponential': 
    theta = np.random.normal(1, 0.25, d)
  if clicking_function == 'combined' or clicking_function == 'squared combined' or clicking_function == 'mixed': 
    theta = np.random.uniform(0, 1, d)
  
  P = np.random.uniform(low=-1, high=1, size=(num_samples, d))

  """
    # Random walk: each step changes slightly from the previous
    for i in range(1, num_samples):
        step = np.random.uniform(low=-0.05, high=0.05, size=(d,))
        P[i] = P[i - 1] + step
        # Optional: Clip to stay in [-1, 1]
        P[i] = np.clip(P[i], -1, 1)
  """
  sensitivity = sim.get_sensitivity()
  
  CTR_obs = np.zeros((num_samples, d))  # Observed CTRs
  X = np.zeros((num_samples, d))
  G = np.zeros((num_samples, d, d))

  if ideal:
    for i in range(num_samples):
      X[i] = sensitivity @ P[i]

      if clicking_function == 'squared':
        CTR_obs[i] = clicking_function_squared(P[i], X[i], theta=theta) #np.random.normal(0.25, 0.1))
        G[i] = sim.get_G(P[i], theta=theta)
      elif clicking_function == 'linear':
        CTR_obs[i] = clicking_function_linear(P[i], X[i])
      elif clicking_function == 'exponential':
        CTR_obs[i] = clicking_function_exponential(P[i], X[i])
      elif clicking_function == 'exponential_half': 
        CTR_obs[i] = clicking_function_exponential_half(P[i], X[i])
      elif clicking_function == 'combined':
        CTR_obs[i] = clicking_function_combined(P[i], X[i], theta=theta)
      elif clicking_function == 'squared combined':
        CTR_obs[i] = clicking_function_squared_combined(P[i], X[i], theta=theta)
      elif clicking_function == 'mixed':
        CTR_obs[i,1:3] = clicking_function_squared_combined(P[i,1:3], X[i,1:3], theta=theta[1:3])
        CTR_obs[i,0] = clicking_function_combined(P[i,0], X[i,0], theta=theta[0])
        CTR_obs[i,3:] = clicking_function_combined(P[i,3:], X[i,3:], theta=theta[3:])

    return (sim, P, CTR_obs, theta, sensitivity)
  
  else:
    for i in range(num_samples):
      if clicking_function == 'combined':
        for j in range(40):
          sim.update(P[i])
        for j in range(20):
          CTR_obs[i] += clicking_function_combined(P[i], sim.update(P[i]), theta= theta)
        CTR_obs[i] /= 20
      
    return (sim, P, CTR_obs, theta, sensitivity)

#sim, P, CTR, G, true_sensitivity = generate_model(num_measurements=1, clicking_function='squared')

#print(G[0])
#print(CTR[0])
