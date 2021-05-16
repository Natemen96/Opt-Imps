#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import os
# os.chdir('./Project2/')
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
import matplotlib.pyplot as plt

def generate_vars(n=100, m = 30):
    """[returns B- nxn, A - mxn, c - mx1]

    Args:
        n (int, optional): [number of columns]. Defaults to 100.
        m (int, optional): [number of rows]. Defaults to 30.

    Returns:
        [np arrays]: [Generated Matrix and Vector]
    """    
    sigma = 1 / (4*n)

    I = np.eye(n)
    K = np.random.normal(scale=sigma, size=(n,n))
    
    A = np.random.uniform(size=(m,n))
    B = I + K + K.T
    c = np.random.uniform(size=(m))
    return A, B, c

def cvxpot_qp(A,c,B):
    """[return x_star]

    Args:
        A ([np.array]): [A - mxn]
        c ([np.array]): [c - mx1]
        B ([np.array]): [B - nxn]

    Returns:
        [type]: [x_star]
    """    
    m = A.shape[0]
    n = A.shape[1]

    q = matrix(np.zeros((n,1)))
    P = matrix(B)
    A = matrix(A)
    b = matrix(c)
    cvx_results= qp(P = P, q = q, A=A, b=b)

    return np.asarray(cvx_results['x']).flatten()

def lagrangian_x(A,c,B,c_k,lam):
    """[returns x_k based on d_x alm solve for x   ]

    Args:
        A ([np.array]): [A - mxn]
        c ([np.array]): [c - mx1]
        B ([np.array]): [B - nxn]
        c_k ([float]): [variable will the following condition
         must be > 0, -> infty, c_{i} <= c_{i+1} ]
        lam ([np.array]): [lam - mx1]

    Returns:
        [x]: [mx1 np.array]
    """    
    leftside = 2*B+(c_k*A.T@A)
    rightside = (c_k*A.T@c) - (A.T@lam)
    x = np.linalg.pinv(leftside)@rightside
    return x.flatten()

def constraint(A,x,c):
    """[returns output of constraint]

    Args:
        A ([np.array]): [A - mxn]
        x ([np.array]): [x - mx1]
        c ([np.array]): [c - mx1]

    Returns:
        [type]: [mx1]
    """    
    return (A@x) - c 

def alm(B,A,c, epsilon=10**(-10)):
    """[uses Augmented Lagrangian method to find an x_k simlar to x_star]

    Args:
        B ([np.array]): [B - nxn]
        A ([np.array]): [A - mxn]
        c ([np.array]): [c - mx1]
        epsilon ([float], optional): [threshold value]. Defaults to 10**(-13).

    Returns:
        [list]: [list of relative error ]
    """    
    x_star = np.asarray(cvxpot_qp(A,c,B*2)).flatten()
    error_i = [] 

    lambda_k = np.zeros(c.shape)
    #c_k must be > 0, -> infinity, c_k_{i} <= c_k_{i+1} 
    c_k =.1
    i = 0 
    while True:
        x_k = lagrangian_x(A,c,B,c_k,lambda_k)
        error = np.linalg.norm(x_k - x_star) /  np.linalg.norm(x_star)
        error_i.append(error)
        stopping = np.linalg.norm(A@x_k - c)
        if stopping <= epsilon:
            return x_k, i, error_i
        
        lambda_k += c_k*constraint(A,x_k,c)

        c_k = 2.0**i

        i += 1

def main():
    """[main function]
    """    
    #a)
    A = np.loadtxt('A_test_1.txt')
    c = np.loadtxt('c_test_1.txt')
    B = np.loadtxt('B_test_1.txt')
    # Q, A, b = generate_vars()

    #b)
    x_star = cvxpot_qp(A,c,B*2)
    print('cvx output')
    print(x_star)

    #c) 
    x_k, i, error_i = alm(B,A,c)
    print()
    print('Augmented Lagrangian method output')
    print(x_k)
    print()
    print('Final Error')
    print(error_i[-1])
    print()
    print('Final Error is below 10^(-6)')
    print(error_i[-1] <= 10**(-6))
    print()
    
    fig = plt.figure()
    plt.title('Relative Error')
    plt.plot(np.arange(len(error_i)), error_i, label = 'Relative Error vs k')
    plt.legend()
    plt.grid(True)
    fig.tight_layout()
    # plt.savefig('pix/Given_sample.png', dpi=fig.dpi)
    plt.show()

if __name__ == "__main__":
    main()