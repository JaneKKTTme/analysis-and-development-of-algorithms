import math
import matplotlib.pyplot as plt
import random

def func1(x):
  return pow(x, 3)

def func2(x):
  return abs(x-0.2)

def func3(x):
  return x * math.sin(1/x)

x1=[]
e=0.001
y1=[]

plt.figure(figsize=(16, 6))

value = 0
while value <= 1:
  x1.append(value)
  y1.append(func1(value))
  value += e
plt.plot(x1, y1, ':')

#exhaustive search
def exhaustive_search(n=1000, e=0.001):
  a=0
  b=1
  x=a
  f_calc, iters = 0, 0
  for i in range(1, n):
    iters+=1
    xi=a+i*(b-a)/n
    if func1(xi) < func1(x):
      f_calc+=2
      x = xi
  return x, f_calc, iters
tmp=exhaustive_search()
plt.plot(tmp[0], func1(tmp[0]), 'yo')

print('EXHAUSTIVE SEARCH for f(x)=x^3')
print('F-calculations: ', tmp[1])
print('The number of iterations: ', tmp[2])

#dichotomy
def dichotomy(e=0.001):
  a=0
  b=1
  x=0
  delta = random.uniform(0.0, e)
  f_calc, iters = 0, 0
  while b-a>e:
    iters+=1
    l = (a+b-delta)/2
    r=(a+b+delta)/2
    if func1(l)<func1(r):
      f_calc+=2
      b=r
      x=l
    else:
      a=l
      x=r
  return x, f_calc, iters
tmp=dichotomy()
plt.plot(tmp[0], func1(tmp[0]), 'r*')
plt.grid(True)

print('DICHOTOMY for f(x)=x^3')
print('F-calculations: ', tmp[1])
print('The number of iterations: ', tmp[2])

PHI = 1.6180339887499
REVERSED_PHI = 1/PHI

#golden section method
def golden_section_search(e=0.001):
  x = 0
  a=0
  b=1
  f_calc, iters = 0, 0
  while (b-a) > e:
    iters+=1
    l = b - (b-a)*REVERSED_PHI
    r = a + (b-a)*REVERSED_PHI
    if func1(l) <= func1(r):
      f_calc+=2
      b = r
      x=l
    else:
      a=l
      x=r
  return x, f_calc, iters

tmp=golden_section_search()
plt.plot(tmp[0], func1(tmp[0]), 'g.')

print('GOLDEN SECTION SEARCH for f(x)=x^3')
print('F-calculations: ', tmp[1])
print('The number of iterations: ', tmp[2])

x2=[]
y2=[]

plt.figure(figsize=(16, 6))

value = 0
while value <= 1:
  x2.append(value)
  y2.append(func2(value))
  value += e
plt.plot(x2, y2, ':')

#exhaustive search
def exhaustive_search(n=1000, e=0.001):
  a=0
  b=1
  x=a
  f_calc, iters = 0, 0
  for i in range(1, n):
    iters+=1
    xi=a+i*(b-a)/n
    if func2(xi) < func2(x):
      f_calc+=2
      x = xi
  return x, f_calc, iters
tmp=exhaustive_search()
plt.plot(tmp[0], func2(tmp[0]), 'yo')

print('EXHAUSTIVE SEARCH for f(x)=|x-0.2|')
print('F-calculations: ', tmp[1])
print('The number of iterations: ', tmp[2])

#dichotomy
def dichotomy(e=0.001):
  a=0
  b=1
  x=0
  delta = random.uniform(0.0, e)
  f_calc, iters = 0, 0
  while b-a>e:
    iters+=1
    l = (a+b-delta)/2
    r=(a+b+delta)/2
    if func2(l)<func2(r):
      f_calc+=2
      b=r
      x=l
    else:
      a=l
      x=r
  return x, f_calc, iters
tmp=dichotomy()
plt.plot(tmp[0], func2(tmp[0]), 'r*')
plt.grid(True)

print('DICHOTOMY for f(x)=|x-0.2|')
print('F-calculations: ', tmp[1])
print('The number of iterations: ', tmp[2])

PHI = 1.6180339887499
REVERSED_PHI = 1/PHI

#golden section method
def golden_section_search(e=0.001):
  x = 0
  a=0
  b=1
  f_calc, iters = 0, 0
  while (b-a) > e:
    iters+=1
    l = b - (b-a)*REVERSED_PHI
    r = a + (b-a)*REVERSED_PHI
    if func2(l) <= func2(r):
      f_calc+=2
      b = r
      x=l
    else:
      a=l
      x=r
  return x, f_calc, iters
tmp=golden_section_search()
plt.plot(tmp[0], func2(tmp[0]), 'g.')

print('GOLDEN SECTION SEARCH for f(x)=|x-0.2|')
print('F-calculations: ', tmp[1])
print('The number of iterations: ', tmp[2])

x3=[]
y3=[]

plt.figure(figsize=(16, 6))

value = 0.01
while value <= 1:
  x3.append(value)
  y3.append(func3(value))
  value += e
plt.plot(x3, y3, ':')

#exhaustive search
def exhaustive_search(n=1000, e=0.001):
  a=0.01
  b=1
  x=a
  f_calc, iters = 0, 0
  for i in range(1, n):
    iters+=1
    xi=a+i*(b-a)/n
    if func3(xi) < func3(x):
      f_calc+=2
      x = xi
  return x, f_calc, iters
tmp=exhaustive_search()
plt.plot(tmp[0], func3(tmp[0]), 'yo')

print('EXHAUSTIVE SEARCH for f(x)=x*sin(1/x)')
print('F-calculations: ', tmp[1])
print('The number of iterations: ', tmp[2])

#dichotomy
def dichotomy(e=0.001):
  a=0.01
  b=1
  x=0
  delta = random.uniform(0.0, e)
  f_calc, iters = 0, 0
  while b-a>e:
    iters+=1
    l = (a+b-delta)/2
    r=(a+b+delta)/2
    if func3(l)<func3(r):
      f_calc+=2
      b=r
      x=l
    else:
      a=l
      x=r
  return x, f_calc, iters
tmp=dichotomy()
plt.plot(tmp[0], func3(tmp[0]), 'r*')
plt.grid(True)

print('DICHOTOMY for f(x)=x*sin(1/x)')
print('F-calculations: ', tmp[1])
print('The number of iterations: ', tmp[2])

PHI = 1.6180339887499
REVERSED_PHI = 1/PHI

#golden section method
def golden_section_search(e=0.001):
  x = 0
  a=0.01
  b=1
  f_calc, iters = 0, 0
  while (b-a) > e:
    iters+=1
    l = b - (b-a)*REVERSED_PHI
    r = a + (b-a)*REVERSED_PHI
    if func3(l) <= func3(r):
      f_calc+=2
      b = r
      x=l
    else:
      a=l
      x=r
  return x, f_calc, iters

tmp=golden_section_search()
plt.plot(tmp[0], func3(tmp[0]), 'g.')

print('GOLDEN SECTION SEARCH for f(x)=x*sin(1/x)')
print('F-calculations: ', tmp[1])
print('The number of iterations: ', tmp[2])


import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize

k = 100

alfa = random.random()
beta = random.random()

def linear_func(a, x, y):
  return a[0] + a[1] * x - y

def linear_lse(a, x, y):
  for i in range(0, len(x)):
    D = ((a[0] + a[1] * x[i] - y[i])**2)
  return D

def rational_func(a, x, y):
    return a[0]/(1+a[1] * x) - y

def rational_lse(a, x, y):
  for i in range(0, len(x)):
    D = ((a[0]/(1 + a[1] * x[i]) - y[i])**2)
  return D

def lse(function, x, a, b, y):
  d = sum([pow(function(x[i], a, b)-y[i], 2) for i in range(k)])
  return d

x = [i/100.0 for i in range(1, k+1)]
y = []

for i in range(1, k+1):
  sigma = random.uniform(-1.0,1.0)
  y.append(alfa*x[k-1] + beta + sigma)

def exhaustive_search(f, x, y):
  a = np.array(np.linspace(0, 1, 100))
  b = np.array(np.linspace(0, 1, 100))
  min = f([0.5, 0], x, y)
  for i in range(len(a)):
    for j in range(len(b)):
      D = f([a[i],b[j]], x, y)
      if min > D:
        min = D
        mass_min = [round(a[i], 2), round(b[j], 2)]
  return mass_min

def MPD1(f, constant, a, b, x, y, eps=0.001):
    n = 0
    while abs(b - a) > eps:
        n += 1
        mid = (a + b) / 2
        if (f([mid + eps, constant], x, y) > f([mid - eps, constant], x, y)):
            b = mid
        else:
            a = mid
    return mid

def MPD2(f, constant, a, b, x, y, eps=0.001):
    n = 0
    while abs(b - a) > eps:
        n += 1
        mid = (a + b) / 2
        if (f([constant, mid + eps], x, y) > f([constant, mid - eps], x, y)):
            b = mid
        else:
            a = mid
    return mid

def coordinate_descent(f, a0, x, y):
  a = np.array([0.32, 0])
  n=0
  while abs(f(a0, x, y) - f([a[0], a[1]], x, y)) >= 0.001:
    n+=1
    a = a0
    a0 = [a0[0]+0.05, a0[1]+0.05]
    for i in range(len(a0)):
      if i == 0:
        constant = a0[1]
        a0[0] = MPD1(f, constant, 0, 1, x, y)
      if i == 1:
        constant = a0[0]
        a0[1] = MPD2(f, constant, 0, 1, x, y)
  return a0

x = np.array(x)
y = np.array(y)
a0 = np.array([1, 1])

res_lsq = least_squares(linear_func, x0=a0, args=(x, y))
res_nm = minimize(linear_lse, a0, args=(x, y),  method='nelder-mead')
res_es = exhaustive_search(linear_lse, x, y)
res_gauss = coordinate_descent(linear_lse, a0, x, y)

print("a = %.2f, b = %.2f" % tuple(res_lsq.x))
print("a = %.2f, b = %.2f" % tuple(res_nm.x))
print("a = ", res_es[0], "b = ", res_es[1])
print("a = ", round(res_gauss[0],2), "b = ", round(res_gauss[1], 2))

f = lambda x: sum([u * v for u, v in zip(res_lsq.x, [1, x])])
f_nm = lambda x: sum([u * v for u, v in zip(res_nm.x, [1, x])])
f_es = lambda x: sum([u * v for u, v in zip(res_es, [1, x])])
f_gauss = lambda x: sum([u * v for u, v in zip(res_gauss, [1, x])])
x_p = np.linspace(min(x), max(x), 20)
y_p = f(x_p)
y_nm = f_nm(x_p)
y_es = f_es(x_p)
y_gauss = f_gauss(x_p)

x1 = np.array(x)
y1 = np.array(y)
a1 = np.array([1, 1])

res_lsq1 = least_squares(rational_func, x0=a1, args=(x1, y1))
res_nm1 = minimize(rational_lse, a1, args=(x1, y1),  method='nelder-mead')
res_es1 = exhaustive_search(rational_lse, x1, y1)
res_gauss1 = coordinate_descent(rational_lse, a1, x1, y1)

print("a = %.2f, b = %.2f" % tuple(res_lsq1.x))
print("a = %.2f, b = %.2f" % tuple(res_nm1.x))
print("a = ", res_es1[0], "b = ", res_es1[1])
print("a = ", round(res_gauss1[0],2), "b = ", round(res_gauss1[1], 2))

f1 = lambda x1: sum([u * v for u, v in zip(res_lsq1.x, [1, x1])])
f_nm1 = lambda x1: sum([u * v for u, v in zip(res_nm1.x, [1, x1])])
f_es1 = lambda x: sum([u * v for u, v in zip(res_es1, [1, x])])
f_gauss1 = lambda x: sum([u * v for u, v in zip(res_gauss1, [1, x])])
x_p1 = np.linspace(min(x1), max(x1), 20)
y_p1 = f(x_p1)
y_nm1 = f_nm1(x_p1)
y_es1 = f_es1(x_p1)
y_gauss1 = f_gauss1(x_p1)

fig, ax = plt.subplots(figsize=(20,5))
plt.plot(x, y, '.r', label='Generated data')
plt.plot(x_p, y_p, 'g', label='Generating line')
plt.plot(x_p, y_es, 'y', label='Exhaustive search')
plt.plot(x_p, y_gauss, 'black', label='Coordinate descent')
plt.plot(x_p, y_nm, 'b', label='Nelder-Mead')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(20,5))
plt.plot(x, y, '.r', label = 'Generated datas')
plt.plot(x_p1, y_p1, 'g', label = 'Generating line')
plt.plot(x_p1, y_es1, 'y', label='Exhaustive search')
plt.plot(x_p1, y_gauss1, 'black', label='Coordinate descent')
plt.plot(x_p1, y_nm1, 'b', label='Nelder-Mead')
ax.legend()
plt.show()