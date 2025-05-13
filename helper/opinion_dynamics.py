import numpy as np
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

## Friedkin-Johnsen model
class FJModel:
  def __init__(self, N, gamma_p, gamma_d, A, x_0):
    self.N = N
    self.gamma_p = np.diag(gamma_p)
    self.gamma_d = np.diag(gamma_d)
    self.A = A
    self.d = x_0
    self.x = x_0

  def update(self, p):
    uncertainty = np.random.normal(0, 0.1, self.N)

    self.x = (np.eye(self.N) - self.gamma_p - self.gamma_d) @ self.A @ self.x + self.gamma_p @ p + self.gamma_d @ self.d #@ np.clip(self.d + uncertainty, -1, 1)
    return self.x
  
  def feedforward(self):
    A_tilde = (np.eye(self.N) - self.gamma_p - self.gamma_d) @ self.A
    inverse = np.linalg.inv(np.eye(self.N) - A_tilde) 
    return -1 * np.linalg.inv((inverse @ self.gamma_p - np.eye(self.N))) @ inverse @ self.gamma_d @ self.d

  def ofo(self, prev_p): ### THere are some mistakes!!!
    nabla = 0.5
    A_tilde = (np.eye(self.N) - self.gamma_p - self.gamma_d) @ self.A
    phi = 0.5 * (np.linalg.inv(np.eye(self.N) - A_tilde) @ self.gamma_p - np.eye(self.N)) @ (self.x - prev_p)
    return np.clip(prev_p - nabla * phi, -1, 1)


def compute_delta(sensitivity, constraints):
  column_sums = np.sum(sensitivity, axis=0)
  max_indices = np.argsort(column_sums)[-constraints:]  # Get indices of top `constraints` columns
  delta = np.zeros(sensitivity.shape[1])
  delta[max_indices] = 1
  print(column_sums, delta)
  return delta

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
    nabla = 0.1
    A_tilde = (np.eye(self.N) - self.gamma) @ self.A
    sensitivity = np.linalg.inv(np.eye(self.N) - A_tilde) @ self.gamma
    
    phi = sensitivity.T @ (self.x - np.ones(self.N))
    p = prev_p - nabla * phi

    p = projection_box(p, (-1, 1))
    p = projection_l1_ball(p, constraint) if constraint is not None else p
    return p
  

  def ofo_sensitivity(self, prev_p, sensitivity):
    nabla = 0.1
    sigma_pe = 0.07

    phi = sensitivity.T @ (self.x - np.ones(self.N))
    p = prev_p - nabla * phi + np.random.normal(0, sigma_pe, self.N)
    p = projection_box(p, (-1, 1))
    return p


  def get_sensitivity(self):
    A_tilde = (np.eye(self.N) - self.gamma) @ self.A
    sensitivity = np.linalg.inv(np.eye(self.N) - A_tilde) @ self.gamma
    return sensitivity

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