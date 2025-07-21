import numpy as np
from helper.opinion_dynamics import DGModel

def clicking_function_squared(p, opinions, theta):
  """Clicking function model: 1 - theta * (opinions - p)**2"""
  return np.ones_like(p) - theta*(opinions - p)**2

def clicking_function_exponential(p, opinions, theta):
  """Clicking function model: exp(-theta * (opinions - p)**2)"""
  return np.exp(-theta*(opinions - p)**2)

def clicking_function_exponential_half(p, opinions):
  """Clicking function model: 0.5 + 0.5*exp(-4 * (opinions - p)**2)"""
  return 0.5*np.ones_like(p) + 0.5*np.exp(-4*(opinions - p)**2)


def generate_model(num_measurements, ideal=True, clicking_function=['squared', 'exponential', 'exponential_half']):
  num_samples = num_measurements
  d = 5  # Dimension of p and x
  A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0, 0.55, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
  
  #A = np.array([[0.15, 0.15, 0.1, 0.2, 0.4],[0.1, 0.45, 0, 0, 0.45],[0.3, 0.05, 0.05, 0, 0.6],[0, 0.4, 0.1, 0.5, 0],[0, 0.3, 0, 0, 0.7]])
  
  x_0 = np.random.uniform(-1, 1, d)
  gamma_p = np.random.uniform(0.01, 0.5, d)
  sim = DGModel(N=d, gamma=gamma_p, A=A, x_0=x_0)
  theta = np.random.normal(0.25, 0.1, d)

  P = np.random.uniform(low=-1, high=1, size=(num_samples, d))
  sensitivity = sim.get_sensitivity()
  
  CTR_obs = np.zeros((num_samples, d))  # Observed CTRs
  X = np.zeros((num_samples, d))
  G = np.zeros((num_samples, d, d))

  if ideal:
    for i in range(num_samples):
      X[i] = sensitivity @ P[i]

      if clicking_function == 'squared':
        CTR_obs[i] = clicking_function_squared(P[i], X[i], theta=theta)
        G[i] = sim.get_G(P[i], np.ones(d)*0.25)

      elif clicking_function == 'exponential':
        CTR_obs[i] = clicking_function_exponential(P[i], X[i], theta=np.ones(d))
      elif clicking_function == 'exponential_half':
        CTR_obs[i] = clicking_function_exponential_half(P[i], X[i])

  return (sim, P, CTR_obs, G, sensitivity)

#sim, P, CTR, G, true_sensitivity = generate_model(num_measurements=1, clicking_function='squared')

#print(G[0])
#print(CTR[0])
