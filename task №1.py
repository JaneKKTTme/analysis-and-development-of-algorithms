import random

ATTEMPTS = 5

n=2000
v_original=[]
v=[]

for i in range(n):
  v_original.append(random.randint(0,100000))
v=v_original
print(v)

import time
import matplotlib.pyplot as plt
from statistics import mean

def const_func(index):
  return 1

try_time=[]
y=[]
trr = []

plt.figure(figsize=(35, 8))

for index in range(n):
  y.append(index+1)
  mid_time=[]
  for j in range(ATTEMPTS):
    start_time = time.time()
    const_func(index)
    end_time = time.time()
    mid_time.append(float(end_time)-float(start_time))
  try_time.append(mean(mid_time))
plt.plot(y, try_time, ls='-', ms=1, lw=0.7, mew=1)
start = time.time()
const_func(1)
end = time.time()
trr = [mean(try_time) for x in range(n)]
plt.plot(y, trr, ls='-', ms=1, mec='r', lw=1, mew=1)
plt.grid(True)

def Task(index):
  y=0
  for k in range(index):
    y+=v[k]
  return y

try_time=[]
trr = []
y=[]

plt.figure(figsize=(30, 7))

for index in range(n):
  y.append(index+1)
  mid_time=[]
  for j in range(ATTEMPTS):
    start_time = time.time()
    Task(index+1)
    end_time = time.time()
    mid_time.append(float(end_time)-float(start_time))
  try_time.append(mean(mid_time))
  trr.append((index+1)/10000000)
plt.plot(y,try_time,"", ls='-', ms=5, lw=2, mec='b', mew=1)
plt.plot(y, trr, ls='-', ms=1, mec='r', lw=1, mew=1)
plt.grid(True)

def prod_func(index):
  y=1
  for k in range(index):
    y=v[k]*y
  return y

try_time=[]
y=[]
trr = []

plt.figure(figsize=(30, 7))

for index in range(n):
  y.append(index+1)
  mid_time=[]
  for j in range(0,5):
    start_time = time.time()
    prod_func(index)
    end_time = time.time()
    mid_time.append(float(end_time)-float(start_time))
  try_time.append(mean(mid_time))
  trr.append(pow((index+1), 2)/10000000000*3.6)
plt.plot(y, try_time,"", ls='-', ms=5, lw=2, mec='b', mew=1)
plt.plot(y, trr, ls='-', ms=1, mec='r', lw=1, mew=1)
plt.grid(True)

import math

def Task(index,x):
  y=0.0
  for k in range(index):
    y=y+float(v[k]*pow(x/10,k+1)) 
  return y

try_time=[]
y=[]
trr = []

plt.figure(figsize=(30, 6))

for index in range(n): #граничное значение перед переполнением, не делать больше
  y.append(index+1)
  mid_time=[]
  for j in range(0,5):
    start_time = time.time()
    Task(index,1.5)
    end_time = time.time()
    mid_time.append(float(end_time)-float(start_time))
  try_time.append(mean(mid_time))
  trr.append((index+1)*math.log((index+1), 2)/23000000)
plt.plot(y, try_time,"", ls='-', ms=5, lw=2, mec='b', mew=1)
plt.plot(y, trr, ls='-', ms=1, mec='r', lw=1, mew=1)
plt.grid(True)

def Task(index,x): 
  y=0.0
  for k in range(index-1, 0, -1):
    y=(y+v[k])*x
  return y

try_time=[]
y=[]
trr = []

plt.figure(figsize=(30, 7))

for index in range(n):
  y.append(index+1)
  mid_time=[]
  for j in range(0,5):
    start_time = time.time()
    Task(index,1.5)
    end_time = time.time()
    mid_time.append(float(end_time)-float(start_time))
  try_time.append(mean(mid_time))
  trr.append((index+1)*math.log((index+1), 2)/85000000)
plt.plot(y, try_time,"", ls='-', ms=5, lw=2, mec='b', mew=1)
plt.plot(y, trr, ls='-', ms=1, mec='r', lw=1, mew=1)
plt.grid(True)

tmp = v

#buble sort
def buble_sort(index):
  n = len(tmp[:index])
  for i in range(n):
    for j in range(0, index-i-1):
      if tmp[j] > tmp[j+1]:
        tmp[j], tmp[j+1] = tmp[j+1], tmp[j]

try_time=[]
y=[]
trr=[]

plt.figure(figsize=(40, 7))

for index in range(1, 500):
  y.append(index)
  mid_time=[]
  for j in range(ATTEMPTS):
    start = time.time()
    buble_sort(index)
    end = time.time()
    mid_time.append(float(end)-float(start))
  try_time.append(mean(mid_time))
  trr.append(pow(index, 2)/15300000)
plt.plot(y, try_time,"", ls='-', ms=5, lw=2, mec='b', mew=1)
plt.plot(y, trr, ls='-', ms=1, mec='r', lw=1, mew=1)
plt.grid(True)

tmp = v

#quick sort
def quicksort(l, r, nums):
  if len(nums) == 1:
    return nums
  if l < r:
    pi = partition(l, r, nums)
    quicksort(l, pi-1, nums)
    quicksort(pi+1, r, nums)
  return nums

def partition(l, r, nums):
  pivot, ptr = nums[r], l
  for i in range(l, r):
    if nums[i] <= pivot:
      nums[i], nums[ptr] = nums[ptr], nums[i]
      ptr += 1
  nums[ptr], nums[r] = nums[r], nums[ptr]
  return ptr


try_time=[]
y=[]
trr=[]

plt.figure(figsize=(40, 7))

for index in range(1, 1000):
  y.append(index)
  mid_time=[]
  for j in range(ATTEMPTS):
    start = time.time()
    quicksort(0, index, tmp)
    end = time.time()
    mid_time.append(float(end)-float(start))
  try_time.append(mean(mid_time))
  trr.append(index*math.log(index, 2)/199000)
plt.plot(y, try_time,"", ls='-', ms=5, lw=2, mec='b', mew=1)
plt.plot(y, trr, ls='-', ms=1, mec='r', lw=1, mew=1)
plt.grid(True)

tmp = v

#timsort
MIN_MERGE = 32

def calcMinRun(n):
    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r

def insertionSort(arr, left, right):
    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1

def merge(arr, l, m, r):
    len1, len2 = m - l + 1, r - m
    left, right = [], []
    for i in range(0, len1):
        left.append(arr[l + i])
    for i in range(0, len2):
        right.append(arr[m + 1 + i])
 
    i, j, k = 0, 0, l

    while i < len1 and j < len2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
 
        else:
            arr[k] = right[j]
            j += 1
 
        k += 1

    while i < len1:
        arr[k] = left[i]
        k += 1
        i += 1

    while j < len2:
        arr[k] = right[j]
        k += 1
        j += 1

def timsort(arr):
    n = len(arr)
    minRun = calcMinRun(n)

    for start in range(0, n, minRun):
        end = min(start + minRun - 1, n - 1)
        insertionSort(arr, start, end)

    size = minRun
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))
            if mid < right:
                merge(arr, left, mid, right)
        size = 2 * size


try_time=[]
y=[]
trr=[]

plt.figure(figsize=(40, 7))

for index in range(1, n+1):
  y.append(index)
  mid_time=[]
  for j in range(ATTEMPTS):
    start = time.time()
    timsort(tmp[:index])
    end = time.time()
    mid_time.append(float(end)-float(start))
  try_time.append(mean(mid_time))
  trr.append((index+1)*math.log(index, 2)/5000000)
plt.plot(y, try_time,"", ls='-', ms=5, lw=2, mec='b', mew=1)
plt.plot(y, trr, ls='-', ms=1, mec='r', lw=1, mew=1)
plt.grid(True)

import numpy
n=400
A=[]
B=[]
for i in range(0,n):
  A.append([0]*n)
  B.append([0]*n)
for i in range(0,n):  
  for j in range(0,n):
    A[i][j]=random.randint(0,100000)
    B[i][j]=random.randint(0,100000)

def Task(index):
  result_matr = [[0 for i in range(index)] for i in range(index)]
  for i in range(index):
    for j in range(index):
      for k in range(index):
        result_matr[i][j] += A[i][k] * B[k][j]
  return result_matr

plt.figure(figsize=(40, 7))

try_time=[]
y=[]
trr=[]

for index in range(1,1000):
  y.append(index)
  mid_time=[]
  for j in range(ATTEMPTS):
    start_time = time.time()
    Task(index)
    end_time = time.time()
    mid_time.append(float(end_time)-float(start_time))
  try_time.append(mean(mid_time))
  trr.append(pow(index, 3)/3900000)

plt.plot(y, try_time, ls='-', ms=5, lw=1, mec='b', mew=1)
plt.plot(y, trr, ls='-', ms=5, lw=3, mec='r', mew=1)

plt.grid(True)