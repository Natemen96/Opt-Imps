#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import os
# os.chdir('./Project2/')
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
import matplotlib.pyplot as plt

def generate_vars(n=100, m = 30):
    """[returns Q- nxn, A - mxn, b - mx1]

    Args:
        n (int, optional): [number of columns]. Defaults to 100.
        m (int, optional): [number of rows]. Defaults to 30.

    Returns:
        [np arrays]: [Generated Matrix and Vector]
    """    
    sigma = 1 / (4*n)

    I = np.eye(n)
    K = np.random.normal(scale=sigma, size=(n,n))
    Q = I + K + K.T
    A = np.random.uniform(size=(m,n))
    b = np.random.uniform(size=(m))
    return Q, A, b

def cvxpot_qp(A,b,Q):
    """[return x_star]

    Args:
        A ([np.array]): [A - mxn]
        b ([np.array]): [b - mx1]
        Q ([np.array]): [Q - nxn]

    Returns:
        [type]: [description]
    """    
    m = A.shape[0]
    n = A.shape[1]

    q = matrix(np.zeros((n,1)))
    P = matrix(Q)
    A = matrix(A)
    b = matrix(b)
    cvx_results= qp(P = P, q = q, A=A, b=b)

    return np.asarray(cvx_results['x']).flatten()

def lagrangian_x(A,b,Q,c,lam):
    """[returns x_k based on d_x alm solve for x   ]

    Args:
        A ([np.array]): [A - mxn]
        b ([np.array]): [b - mx1]
        Q ([np.array]): [Q - nxn]
        c ([float]): [variable will the following condition
         must be > 0, -> infty, c_{i} <= c_{i+1} ]
        lam ([np.array]): [lam - mx1]

    Returns:
        [x]: [mx1 np.array]
    """    
    leftside = 2*Q+(c*A.T@A)
    rightside = (c*A.T@b) - (A.T@lam)
    x = np.linalg.pinv(leftside)@rightside
    return x.flatten()

def constraint(A,x,b):
    """[returns output of constraint]

    Args:
        A ([np.array]): [A - mxn]
        x ([np.array]): [x - mx1]
        b ([np.array]): [b - mx1]

    Returns:
        [type]: [mx1]
    """    
    return (A@x) - b 

def alm(Q,A,b, epsilon=10**(-10)):
    """[uses Augmented Lagrangian method to find an x_k simlar to x_star]

    Args:
        Q ([np.array]): [Q - nxn]
        A ([np.array]): [A - mxn]
        b ([np.array]): [b - mx1]
        epsilon ([float], optional): [threshold value]. Defaults to 10**(-13).

    Returns:
        [list]: [list of relative error ]
    """    
    x_star = np.asarray(cvxpot_qp(A,b,Q*2)).flatten()
    error_i = [] 

    lambda_k = np.zeros(b.shape)
    #c_k must be > 0, -> infinity, c_k_{i} <= c_k_{i+1} 
    c_k =.1
    i = 0 
    while True:
        x_k = lagrangian_x(A,b,Q,c_k,lambda_k)
        error = np.linalg.norm(x_k - x_star) /  np.linalg.norm(x_star)
        error_i.append(error)
        stopping = np.linalg.norm(A@x_k - b)
        if stopping <= epsilon:
            return x_k, i, error_i
        
        lambda_k += c_k*constraint(A,x_k,b)

        c_k = 2.0**i

        i += 1

def main():
    """[main function]
    """    
    #a)
    A = np.loadtxt('A_test_1.txt')
    b = np.loadtxt('b_test_1.txt')
    Q = np.loadtxt('Q_test_1.txt')
    # Q, A, b = generate_vars()

    #b)
    x_star = cvxpot_qp(A,b,Q*2)
    print('cvx output')
    print(x_star)

    #c) 
    x_k, i, error_i = alm(Q,A,b)
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