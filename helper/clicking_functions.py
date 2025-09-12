import numpy as np

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

def minimize_squared(x, d, p, ctr):
    num_rounds = np.shape(p)[0]

    sensitivity = x[:d*d].reshape((d, d), order='C')
    theta = x[d*d:]

    x = sensitivity @ p.T

    return np.ones(num_rounds * d) - np.tile(theta, num_rounds).T*(x.flatten(order='F') - p.flatten(order='C'))**2 - ctr.flatten(order='C')

def minimize_squared_one_theta(x, d, p, ctr):
    num_rounds = np.shape(p)[0]

    sensitivity = x[:d*d].reshape((d, d), order='C')
    theta = x[-1]

    x = sensitivity @ p.T

    return np.ones(num_rounds * d) - theta*(x.flatten(order='F') - p.flatten(order='C'))**2 - ctr.flatten(order='C')

def minimize_exponential(x, d, p, ctr):
    num_rounds = np.shape(p)[0]

    sensitivity = x[:d*d].reshape((d, d), order='C')
    theta = x[d*d:]
    x = sensitivity @ p.T

    row_stochastic = np.sum(sensitivity, axis=1) - np.ones(d)
    clicking = np.exp(-np.tile(theta, num_rounds).T*(x.flatten(order='F') - p.flatten(order='C'))**2) - ctr.flatten(order='C')

    return np.concatenate((clicking, row_stochastic))

def minimize_combined(x, d, p, ctr):
    num_rounds = np.shape(p)[0]

    sensitivity = x[:d*d].reshape((d, d), order='C')
    theta = x[d*d:]
    x = sensitivity @ p.T

    row_stochastic = np.sum(sensitivity, axis=1) - np.ones(d)
    clicking = 0.5*(1 - np.tile(theta, num_rounds).T) + np.tile(theta, num_rounds).T*np.exp(-4*(x.flatten(order='F') - p.flatten(order='C'))**2) - ctr.flatten(order='C')

    return np.concatenate((clicking, row_stochastic))

def minimize_combined_scalar(x, d, p, ctr):
    num_rounds = np.shape(p)[0]

    sensitivity = x[:d*d].reshape((d, d), order='C')
    theta = x[d*d:]
    x = sensitivity @ p.T

    #row_stochastic = np.sum(sensitivity, axis=1) - np.ones(d)
    clicking = 0.5*(1 - np.tile(theta, num_rounds).T) + np.tile(theta, num_rounds).T*np.exp(-4*(x.flatten(order='F') - p.flatten(order='C'))**2) - ctr.flatten(order='C')

    return np.linalg.norm(clicking, ord=1)**2

def minimize_exponential_fixed_theta(x, d, p, ctr):
    num_rounds = np.shape(p)[0]

    sensitivity = x[:d*d].reshape((d, d), order='C')
    x = sensitivity @ p.T

    row_stochastic = np.sum(sensitivity, axis=1) - np.ones(d)
    clicking = np.exp(-np.ones(d*num_rounds).T*(x.flatten(order='F') - p.flatten(order='C'))**2) - ctr.flatten(order='C')
    #print(np.concatenate((clicking, row_stochastic)))
    return np.concatenate((clicking, row_stochastic))
