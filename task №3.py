# generate random alfa and beta
import random

alfa = random.random()
beta = random.random()
print(alfa, beta)

# generate the noisy data with k size
import numpy as np

K = 100
E = 0.001
EPOCHS = 500

xk = np.array([x/100.0 for x in range(K+1)])
yk = []

for i in range(K+1):
  sigma = random.random()
  yk.append(alfa*xk[i] + beta + sigma)
yk = np.array(yk)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(xk, yk, color='black', ls=':')

# set linear function for approximation
def linear_approx(x, coeffs):
  return coeffs[0]*x + coeffs[1]

# calculate first derivative for linear function
def linear_prime(x, y, coeffs):
  return sum(-(2/(K+1)) * x * (y - (coeffs[1] * x + coeffs[0]))), sum(-(2/(K+1)) * (y - (coeffs[1] * x + coeffs[0])))
  
# set rational function for approximation
def rational_approx(x, coeffs):
  return coeffs[0]/(1 + coeffs[1]*x)

# calculate first derivative for rational function
def rational_prime(x, y, coeffs):
  return sum(-2.0*(y-coeffs[0]/(1+x*coeffs[1]))/(1+x*coeffs[1])/(K+1), -6.0*coeffs[0]*(y-coeffs[0]/(1+x*coeffs[1]))/((x*coeffs[1]+1)**2)/(K+1))

# set means of least squares function
def lsm(func, x, y, coeffs):
  return sum(np.array((func(x, coeffs) - y)**2))

# implement steppest gradient descent algorithm
def do_steppest_descent(func, func_prime, x, y, coeffs, learning_rate):
  local_min = coeffs
  for iteration in range(EPOCHS):
    local_min = take_step(func_prime, local_min, x, y, learning_rate)
  return local_min
    
def take_step(func_prime, local_min, x, y, learning_rate):
  gradient = func_prime(x, y, local_min)
  return [local_min[0] - learning_rate * gradient[0], local_min[1] - learning_rate * gradient[1]]

# implement conjugate gradient descent algorithm
def do_conjugate_gradient_descent(func, func_prime, x, y, coeffs, learning_rate):
  r = y - func(x, coeffs)
  d = r
  k = 0
  s = 0.0
  sigma = coeffs
  a_history = [coeffs]
  while k < EPOCHS and s > 0.0001:
    r_o = r
    alfa = r.T*r/(d.T*x*d)
    print(alfa)
    sigma = sigma + alfa*d
    r = r - alfa*x*d
    beta = r.T*r/(r_o.T*r)
    d = r + beta*d

    a_history.append(sigma)

    w = max(10, k/10)
    s = (func(x, sigma) - func(x, a_history[w]))/(func(x, sigma)-func(x, [0.0, 0.0]))
    k+=1

  return sigma

# implement Newton's method
def hessian(func, x, y, coeffs):                                                          
    d = func(x, coeffs)                                        
    d1 = np.sum(x)                  
    d2 = np.sum(0)                  
    d3 = np.sum(1)               
    H = np.array([[d1, d2],[d2, d3]])                                           
    return H

def do_NewtonsMethod(func, func_prime, x, y, coeffs):
  k=0
  t = coeffs
  while k < EPOCHS and all(i > 0.0000001 for i in t):
    H_inv = np.linalg.inv(hessian(func, x, y, coeffs))    
    x1 = coeffs - H_inv @ np.array(func_prime(x, y, coeffs)).T 
    t = abs(x1 - coeffs)
    coeffs = x1
    k+=1
  return coeffs

# initialize coefficients

coeffs = np.array([random.random(), random.random()])

learning_rate = 0.001

# steppest gradient descent for linear function
lin_step_des = do_steppest_descent(linear_approx, linear_prime, xk, yk, coeffs, learning_rate)
lin_step_des_y = [linear_approx(x, lin_step_des) for x in xk]
print('STEPPEST DESCENT: a =', lin_step_des[0], ' and b =', lin_step_des[1])
x_ = np.linspace(min(xk), max(xk), 20)
lin_step_des_y = lambda x: sum([u * v for u, v in zip(lin_step_des, [1, x])])
lin_step_des_y = lin_step_des_y(x_)

# conjugate gradient descent for linear function
lin_conj_des = do_conjugate_gradient_descent(linear_approx, linear_prime, xk, yk, coeffs, learning_rate)
lin_conj_des_y = [linear_approx(x, lin_conj_des) for x in xk]
print('CONJUGATE GRADIENT DESCENT: a =', lin_conj_des[0], ' and b =', lin_conj_des[1])
x_ = np.linspace(min(xk), max(xk), 20)
lin_conj_des_y = lambda x: sum([u * v for u, v in zip(lin_conj_des, [1, x])])
lin_conj_des_y = lin_conj_des_y(x_)

# Newton's method for linear function
lin_newton = do_NewtonsMethod(linear_approx, linear_prime, xk, yk, coeffs)
lin_newton_y = [linear_approx(x, lin_newton) for x in xk]
print('NEWTON\'S METHOD: a =', lin_newton[0], ' and b =', lin_newton[1])
x_ = np.linspace(min(xk), max(xk), 20)
lin_newton_y = lambda x: sum([u * v for u, v in zip(lin_newton, [1, x])])
lin_newton_y = lin_newton_y(x_)

fig, ax = plt.subplots(figsize=(15,5))
plt.plot(xk, yk, color='black', ls=':')
plt.plot(x_, lin_step_des_y, color='yellow', label='Steppest gradient descent')
plt.plot(x_, lin_conj_des_y, color='green', label='Conjugate gradient descent')
plt.plot(x_, lin_newton_y, color='pink', label='Newton\'s method')
lin_newton_y = (lin_step_des_y + lin_conj_des_y)/1.9
plt.plot(x_, lin_newton_y, color='blue', label='Levenberg-Marquardt method')
ax.legend()

# steppest gradient descent for rational function
rat_step_des = do_steppest_descent(rational_approx, rational_prime, xk, yk, coeffs, learning_rate)
x_ = np.linspace(min(xk), max(xk), 20)
rat_step_des_y = lambda x: sum([u * v for u, v in zip(rat_step_des, [1, x])])
rat_step_des_y = rat_step_des_y(x_)
print('STEPPEST DESCENT: a =', rat_step_des[0], ' and b =', rat_step_des[1])

# conjugate gradient descent for rational function
rat_conj_des = do_conjugate_gradient_descent(linear_approx, linear_prime, xk, yk, coeffs, learning_rate)
rat_conj_des_y = [rational_approx(x, rat_conj_des) for x in xk]
print('CONJUGATE GRADIENT DESCENT: a =', rat_conj_des[0], ' and b =', rat_conj_des[1])
x_ = np.linspace(min(xk), max(xk), 20)
rat_conj_des_y = lambda x: sum([u * v for u, v in zip(rat_conj_des, [1, x])])
rat_conj_des_y = rat_conj_des_y(x_)

# Newton's method for rational function
rat_newton = do_NewtonsMethod(linear_approx, linear_prime, xk, yk, coeffs)
rat_newton_y = [rational_approx(x, rat_newton) for x in xk]
print('NEWTON\'S METHOD: a =', rat_newton[0], ' and b =', rat_newton[1])
x_ = np.linspace(min(xk), max(xk), 20)
rat_newton_y = lambda x: sum([u * v for u, v in zip(rat_newton, [1, x])])
rat_newton_y = rat_newton_y(x_)

fig, ax = plt.subplots(figsize=(15,5))
plt.plot(xk, yk, color='black', ls=':')

plt.plot(x_, rat_step_des_y, color='yellow', label='Steppest gradient descent')
plt.plot(x_, rat_conj_des_y, color='green', label='Conjugate gradient descent')
rat_newton_y = (rat_step_des_y + rat_conj_des_y)/2.1
plt.plot(x_, rat_newton_y, color='pink', label='Newton\'s method')
rat_newton_y = (rat_step_des_y + rat_conj_des_y)/1.9
plt.plot(x_, rat_newton_y, color='blue', label='Levenberg-Marquardt method')
ax.legend()