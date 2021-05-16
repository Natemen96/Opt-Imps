#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as opt

B = np.loadtxt('B.txt')
c = np.loadtxt('c.txt')
d = np.loadtxt('d.txt')

def gen_B_c_d(n=5):
    """[Function created from Q_b_c.py to quickly test program]

    Args:
        n (int, optional): [size of matrix Q (n,n) and vector b (n,)]. Defaults to 5.

    Returns:
        [B,c,d]: [(5,5) numpy P.D matrix, (5,1) numpy vector, () numpy scaler]
    """    
    Q=np.random.random((n,n))
    B=Q.dot(Q.T)+0.1*np.identity(n)
    c=np.random.random(n)
    d=np.random.random(1)
    return B, c, d

def is_pos_def(x):
    """[Returns True if matrix is positive definite]

    Args:
        x ([numpy matrix]): [matrix to be tested]

    Returns:
        [boolean]: [True if matrix is positive definite, false otherwise]
    """    
    return np.all(np.linalg.eigvals(x) > 0)

# B,c,d = gen_B_c_d() #uncomment for testing

def f_x(x ,Q = B, b =c, c = d):
    """[Mathematical Function of Interest: f(x) = x^T Q x + b^T x + c]

    Args:
        x ([numpy vector]): [(5,1) numpy Vector]
        Q ([numpy matrix], optional): [(5,5) numpy P.D Matrix]. Defaults to Q.
        b ([numpy vector], optional): [(5,1) numpy vector]. Defaults to b.
        c ([numpy scaler], optional): [() numpy scaler]. Defaults to c.

    Returns:
        [numpy scaler]: [() numpy scaler]
    """    
    return x.T@Q@x + b.T@x + c 

def nabla_f_x(x ,Q = B, b =c):
    r"""[Gradient of Mathematical Function of Interest: \nabla f(x) = 2Qx +b ]

    Args:
        x ([numpy vector]): [(5,1) numpy vector]
        Q ([numpy matrix], optional): [(5,5) numpy P.D Matrix]. Defaults to Q.
        b ([numpy vector], optional): [(5,1) numpy vector]. Defaults to b.

    Returns:
        [numpy vector]: [(5,1) numpy vector]
    """    

    return 2*Q@x + b

def get_alpha_k(x_k, Q, method = 0 ):
    """[Provides 3 different ways of finding alpha]

    Args:
        x_k ([numpy vector]): [(5,1) numpy vector - Take your best guess ]
        method (int, optional): [0 - Line search, 1 - Armijo method, otherwise - Steepest descent method]. Defaults to 0.
        Q ([numpy matrix], optional): [(5,5) numpy P.D Matrix]. Defaults to Q.

    Returns:
        [float64]: [desired alpha value]
    """  

    def keep_in_bound(x, lower = 0, upper =1):
        """[Helper function for variable with open bounds (0,1) by default]

        Args:
            x ([scaler]): [value that need to be bounded]
            lower (int, optional): [scaler]. Defaults to 0.
            upper (int, optional): [scaler]. Defaults to 1.

        Returns:
            [scaler]: [new bounded value]
        """        
        
        if x < lower or x > upper: 
            x = lower+np.random.rand(1)*upper
            x = keep_in_bound(x, lower=lower, upper=upper) 
            return x 
        else:
            return x

    def line_search(alpha):
        """[Implementation of a Line Search]

        Args:
            alpha ([scaler]): [a potential alpha_k]

        Returns:
            [scaler]: [f_x(x_k - alpha* nabla_f_x(x_k))]
        """        
        return f_x(x_k - alpha* nabla_f_x(x_k))
    
    def armijo_rule(alpha, gamma, beta):
        """[Implementation of a Armijo Rule]

        Args:
            alpha ([scaler]): [alpha to be tested, should be between (0,1)]
            gamma ([scaler]): [should be between (0,1)]
            beta ([scaler]): [should be between (0,1)]

        Returns:
            [scaler]: [alpha that passed Armijo's rule]
        """  
        gamma = keep_in_bound(gamma)
        beta = keep_in_bound(beta) 

        while True:
            if line_search(alpha) <= f_x(x_k) - gamma*alpha * np.linalg.norm(nabla_f_x(x_k))**2:
                return alpha
            else: 
                alpha = beta * alpha
    def steepest_descent():
        r"""[Implementation of a Steppest Descent: a_k = \frac{\nabla f(x_k)^T \nabla f(x_k)}{2 \nabla f(x_k)^T Q \nabla f(x_k)}]

        Returns:
            [type]: [desired alpha]
        """ 
        num = nabla_f_x(x_k).T@nabla_f_x(x_k)
        dom = 2*nabla_f_x(x_k).T@Q@nabla_f_x(x_k)
        alpha = num/dom 
        return alpha

    if method == 0:
        alpha = np.random.rand(1000)
        k = np.argmin(np.vectorize(line_search)(alpha))
        alpha_k = alpha[k]
    elif method == 1:
        alpha = np.random.rand(1)[0]
        alpha_k = armijo_rule(alpha, np.random.rand(1), np.random.rand(1))
    else:
        alpha_k = steepest_descent()
    
    return alpha_k

def method_identifer(method):
    """[helper function for identifying the method being used]

    Args:
        method ([int]): [0 - Line search, 1 - Armijo method, otherwise - Steepest descent method]
    """    

    if method == 0: 
        print('GD - Line_Search')
    elif method == 1: 
        print('GD - Armijo Method')
    else: 
        print('GD - Steepest Descent')

def gradient_descent(x_k, B, method = 0, epsilon = 10**(-6)):
    """[return gradient descent's x_k, f_x(x_k), epsilon, and num of interations (k)]

    Args:
        x_k ([numpy vector]): [(5,1) numpy vector - Take your best guess]
        method (int, optional): [0 - Line search, 1 - Armijo method, otherwise - Steepest descent method]. Defaults to 0.
        epsilon (float, optional): [error tolerance]. Defaults to 10**(-6).

    Returns:
        [numpy vector, numpy scaler, float, int]: [x_k - approximate x_star, f_x(x_k) - approximate min of f(x) , epsilon - error tolerance, i - iterations]
    """    
    method_identifer(method)

    if np.linalg.norm(x_k) <= epsilon:
        print(np.linalg.norm(nabla_f_x(x_k)))
        print('Epsilon {} is too high or Lucky Guess!'.format(epsilon))
        print('x_k : {}'.format(x_k))
        print('f(x_k) : {}'.format(f_x(x_k))) 
        print('Epsilon: {}'.format(epsilon))
        print('k: {}'.format(0))
        print()
        return x_k, f_x(x_k), epsilon, 0
    
    i = 0
    while True:     
        alpha_k = get_alpha_k(x_k, B, method= method)
        x_k = x_k - alpha_k*nabla_f_x(x_k)
        
        error =np.linalg.norm(nabla_f_x(x_k))
        i = i+1
        # if i % 100 == 0:
        #     print('{} iterations so far'.format(i))
        if error <= epsilon:
            print('x_k : {}'.format(x_k))
            print('f(x_k) : {}'.format(f_x(x_k))) 
            print('Epsilon: {}'.format(epsilon))
            print('k: {}'.format(i))
            print()
            return x_k, f_x(x_k), epsilon, i


def reasonable_error(x_k, x_star, method = 0):
    """[summary]

    Args:
        x_k ([numpy vector]): [x_star approximation]
        x_star ([numpy vector]): [min x or reference point]
        method ([int], optional): [0 - Line search, 1 - Armijo method, otherwise - Steepest descent method]. Defaults to 0.
    """    
    method_identifer(method)
    final_error = abs(f_x(x_k) - f_x(x_star))

    if final_error < 10**(-6):
        print(final_error)
        print("Within Reasonable Approximation Error")
        print("^_^")
        print()
    else: 
        print(final_error)
        print("Not Within Reasonable Approximation Error")
        print(">.<")
        print()

def main():
    """[Main function]
    """    


    #a)
    epsilon = 10**(-6)
    x_k0, f_x_k, epsilon, numofinter = gradient_descent(c, B, method =0, epsilon=epsilon) 

    x_k1, f_x_k, epsilon, numofinter = gradient_descent(c, B, method =1, epsilon=epsilon) 

    x_k2, f_x_k, epsilon, numofinter = gradient_descent(c, B, method =2, epsilon=epsilon)

    #b) 
    x_l = -np.linalg.inv(B)@c*(.5) 
    f_l = f_x(x_l)

    print('x^*_l : {}'.format(x_l))
    print('f(x^*_l) : {}'.format(f_l))
    print()

    #c)
    res  = opt.minimize(f_x, c)
    x_star = res.x
    f_star = f_x(x_star)

    print('x^* : {}'.format(x_star))
    print('f(x^*) : {}'.format(f_star))
    print()

    #d.)
    reasonable_error(x_k0, x_star, method = 0)
    # reasonable_error(x_k1, x_star, method = 1)
    reasonable_error(x_k2, x_star, method = 2)

if __name__ == "__main__":
    main()