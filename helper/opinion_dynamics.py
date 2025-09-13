import numpy as np
from scipy.optimize import minimize
from jaxopt.projection import projection_l1_ball, projection_box

def generate_adjacency_matrix(n):
    A = np.random.rand(n, n) * (np.random.rand(n, n) > 0.5)
    np.fill_diagonal(A, 1.0)  # Set diagonal entries to 1
    for it, row in enumerate(A):
        row_sums = row.sum()
        if row_sums == 0:
            A[it][it] = 1
            row_sums = 1
        A[it] = A[it] / row_sums
    return A


def projection(p, kappa):
  d = len(p)
    
  # Objective: minimize ||z - p||^2
  def obj(z):
    return 0.5 * np.sum((z - p)**2)
    
  # Constraints
  cons = []
    
  # l1 norm constraint
  cons.append({'type': 'ineq', 'fun': lambda z: kappa - np.sum(np.abs(z))})
    
  # Bounds 0 <= z_i <= 1
  bounds = [(0,1)] * d
    
  # Solve
  result = minimize(obj, np.clip(p, 0, 1), bounds=bounds, constraints=cons, method='SLSQP')
  #print("Projection success:", result.success, "Message:", result.message)
  return result.x

## Friedkin-Johnsen model
class FJModel:
  def __init__(self, N, gamma_p, gamma_d, A, x_0, d):
    self.N = N
    self.gamma_p = np.diag(gamma_p)
    self.gamma_d = np.diag(gamma_d)
    self.A = A
    self.d = d
    #self.d = x_0
    self.x = x_0

  def update(self, p):
    uncertainty = np.random.normal(0, 0.1, self.N)

    self.x = (np.eye(self.N) - self.gamma_p - self.gamma_d) @ self.A @ self.x + self.gamma_p @ p + self.gamma_d @ self.d #@ np.clip(self.d + uncertainty, -1, 1)
    return self.x
  
  def get_sensitivity(self):
    A_tilde = (np.eye(self.N) - self.gamma_p - self.gamma_d) @ self.A
    sensitivity = np.linalg.inv(np.eye(self.N) - A_tilde) @ self.gamma_p
    return sensitivity
  
  def ofo(self, prev_p, constraint=None):
    eta = 0.075
    A_tilde = (np.eye(self.N) - self.gamma_p - self.gamma_d) @ self.A
    sensitivity = np.linalg.inv(np.eye(self.N) - A_tilde) @ self.gamma_p
    
    phi = sensitivity.T @ (-1*np.ones(self.N))
    p = prev_p - eta * phi

    #p = projection_box(p, (-1, 1))
    #p = projection_l1_ball(p, constraint) if constraint is not None else p
    return projection(p, kappa=constraint)
  

  def ofo_sensitivity(self, prev_p, sensitivity, constraint=None):
    eta = 0.075
    sigma_pe = 0.07

    phi = sensitivity.T @ (-1*np.ones(self.N))
    p = prev_p - eta * phi + np.random.normal(0, sigma_pe, self.N)
    
    #p = projection_box(p, (-1, 1))
    #p = projection_l1_ball(p, constraint) if constraint is not None else p
    return projection(p, kappa=constraint)





class DGModel:
  def __init__(self, N, gamma, A, x_0):
    self.N = N
    self.gamma = np.diag(gamma)
    self.A = A
    self.x = x_0
  
  def update(self, p):
    self.x = (np.eye(self.N) - self.gamma) @ self.A @ self.x + self.gamma @ p
    return self.x

  
  def ofo(self, prev_p, constraint=None):
    eta = 0.075
    A_tilde = (np.eye(self.N) - self.gamma) @ self.A
    sensitivity = np.linalg.inv(np.eye(self.N) - A_tilde) @ self.gamma
    
    phi = sensitivity.T @ (-1*np.ones(self.N))
    p = prev_p - eta * phi

    return projection(p, kappa=constraint)
  

  def ofo_sensitivity(self, prev_p, sensitivity, constraint=None):
    eta = 0.075
    sigma_pe = 0.07

    phi = sensitivity.T @ (-1*np.ones(self.N))
    p = prev_p - eta * phi + np.random.normal(0, sigma_pe, self.N)
    
    return projection(p, kappa=constraint)
  

  
  def evolve_A(self):
    perturbation = np.random.normal(0, 0.01, (self.N, self.N)) * (np.random.rand(self.N, self.N) > 0.95)
    self.A = self.A + perturbation
    self.A = np.clip(self.A, 0, None)  # Ensure non-negativity
    for it, row in enumerate(self.A):
        row_sums = row.sum()
        if row_sums == 0:
            self.A[it][it] = 1
            row_sums = 1
        self.A[it] = self.A[it] / row_sums

  def flip_A(self):
    i = 0
    j = 2
    self.A[:, [i, j]] = self.A[:, [j, i]]

  def get_sensitivity(self):
    A_tilde = (np.eye(self.N) - self.gamma) @ self.A
    sensitivity = np.linalg.inv(np.eye(self.N) - A_tilde) @ self.gamma
    return sensitivity
  
  def get_opinion(self):
    return self.x
  
  def get_CR_sensitivity(self, p):
    sensitivity = self.get_sensitivity()
    diag_Sp = np.diag(sensitivity @ p)
    diag_p = np.diag(p)
    return 0.5 * (diag_Sp + diag_p @ sensitivity)
  
  def get_G(self, p, theta):
    sensitivity = self.get_sensitivity()
    x = sensitivity @ p
    diag = np.diag(2 * theta * (x - p))
    return (- diag + diag @ sensitivity)
  
  def ofo_milp(self, prev_p, constraint=None):
    nabla = 0.1
    A_tilde = (np.eye(self.N) - self.gamma) @ self.A
    sensitivity = np.linalg.inv(np.eye(self.N) - A_tilde) @ self.gamma
    
    phi = sensitivity.T @ (self.x - np.ones(self.N))
    p = prev_p - nabla * phi

    p = projection_box(p, (-1, 1))

    if constraint is None:
      p = np.round(p)
    else:
      binary_p = np.zeros(self.N)
      top_indices = np.argpartition(p, -constraint)[-constraint:]
      binary_p[top_indices] = 1
      p = binary_p
      
    return p


  """
  def ofo(self, prev_p, constraint=None):
    nabla = 0.1
    A_tilde = (np.eye(self.N) - self.gamma) @ self.A
    sensitivity = np.linalg.inv(np.eye(self.N) - A_tilde) @ self.gamma
    
    delta = np.ones(self.N)
    if constraint is not None:
      delta = compute_delta(sensitivity, constraint)

    phi = 2 * (np.linalg.inv(np.eye(self.N) - A_tilde) @ np.diag(delta) @ self.gamma).T @ (self.x - np.ones(self.N))
    return np.clip(prev_p - nabla * phi, -1, 1), delta
  """

class Delta_DGModel:
  def __init__(self, N, gamma, A, x_0, delta=np.ones(5)):
    self.N = N
    self.gamma = np.diag(gamma)
    self.A = A
    self.x = x_0
    self.delta = np.diag(delta)
  
  def update(self):
    self.x = (np.eye(self.N) - self.delta @ self.gamma) @ self.A @ self.x + self.delta @ self.gamma @ np.ones(self.N)
    return self.x