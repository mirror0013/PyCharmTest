# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 00:14:46 2020

@author: jd
"""

import sys

import numpy as np
from scipy.linalg import norm, solve, eigvals
import datetime as dtime

filename = ["Matrix", "pyMatrix", "Matrix100", "pyMatrix100"]
max_iter = [1000000, 1000000, 1000000, 1000000]
dftau = dict(zip(filename, [0.001, 0.001, 0.001, 0.001]))
dfname = dict(zip(filename, max_iter))
Epsilon = 1e-10
for fname in dfname.keys():
    with open(fname + ".txt", "r") as f:
        lines = f.readlines()
    sys.stdout = open(fname + "-py-output.txt", "w")
    n = int(lines[0].strip().split()[0])
    print(n)
    ll = [[float(l) for l in line.strip().split()] for line in lines[1:n + 1]]
    print(len(ll))
    A = np.array(ll)
    print(np.linalg.det(A))
    print(eigvals(A))
    print(f"Matrix shape = {A.shape}")
    print(" Matrix A:\n", A)
    print(f"Matrix rank = {np.linalg.matrix_rank(A)}")
    b = np.array([float(l) for l in lines[n + 2].strip().split()])
    print(f"Length of b = {b.shape}")
    print(" Vector b:\n", b)
    print("\nSolve = \n", *solve(A, b))
    try:
        x = np.array([float(l) for l in lines[n + 4].strip().split()])
    except:
        x = []
    if not len(x):
        x = np.zeros_like(b)
        k = np.arange(n) + 1
        # x = np.cos((k * 2 - 1) * (np.pi / n / 2))
    print(f"Length of x0: {x.shape}")
    print(" Vector x0:\n", x)
    SolveFound = True
    time_elapsed = dtime.datetime.now()
    tau = dftau[fname]
    print("tau = ", tau)
    ksi = np.min(A) / np.max(A)
    ro0 = (1 - ksi) / (1 + ksi)
    tau = tau / (1 + ro0 * x)
    print("tau = ", tau)
    R0 = norm(b, ord=np.inf)
    print("norm(b) = ", R0)
    xnorma = 1.0
    xnormamin = 1.0
    xsolve = np.copy(x)
    for i in range(dfname[fname]):
        print("\nIteration #", i)
        R = b - A.dot(x)
        try:
            xnorma = norm(R, ord=np.inf) / R0
            print("norm(b - A * x) / norm(b) = ", xnorma)
        except:
            SolveFound = False
            break
        if xnorma < 0:
            SolveFound = False
            break
        if (xnormamin > xnorma):
            xnormamin = xnorma
            xsolve = np.copy(x)
        if xnormamin < Epsilon:
            break
        x += R * tau
        # print(np.sum(x))
    else:
        SolveFound = False
    print("\n xnorma = ", xnorma)
    print("\nElapsed time: ", dtime.datetime.now() - time_elapsed)
    if SolveFound:
        print("\nx = \n", *x)
    else:
        print("\nSolve wasn't found!\n")
        print("best xnorma = ", xnormamin)
        print("\nbest x = \n", *xsolve)
    print("\nSolve = \n", *solve(A, b))
    sys.stdout.close()
