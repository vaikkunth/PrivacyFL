import scipy.integrate as integrate
import numpy as np

'''
This is a helper function for generating the probability density function for truncated laplacee distribution.
This function processes A and B to be the desired value. If the input A, B value is not appropriate for the desired 
epsilon, we default to the values for symmetric A, B values. 
'''
def process(epsilon, global_sensitivity, A, B):
    lam = 1 / epsilon
    A_ = lam * np.log(2 + (1 - global_sensitivity) / global_sensitivity * np.exp(-B / lam) - 1 / global_sensitivity * np.exp(-(B - 1) / lam))
    if abs(B) < abs(A_):
        return A_, B
    B_ = -lam * np.log(2 + (1 - global_sensitivity)/global_sensitivity * np.exp(A/lam) - 1/global_sensitivity * np.exp((B+1)/lam))
    if abs(A) < abs(B_):
        return A, B_
    A_ = global_sensitivity / epsilon * np.log(1 + (np.exp(epsilon) - 1)/(2 * global_sensitivity))
    B_ = - A_
    return A_, B_

'''
Given privacy parameters epsilon, global_sensitivity, and scale parameters A, and B, return a function that is the probability 
density function for the truncated laplace distribution.
'''
def truncated_laplace(epsilon, global_sensitivity, A, B):
    lam = 1 / epsilon
    M = 1 / (lam * (2 - np.exp(A / lam) - np.exp(-B / lam)))
    return lambda x: M * np.exp(-abs(x) / lam)

'''
Given epsilon, global_sensitivity, A, B, return the L1 cost for the truncated laplacian mechanism.
'''
def truncated_laplace_L1_eval(epsilon, global_sensitivity, A, B):
    return integrate.quad(lambda x: abs(x) * truncated_laplace(epsilon, global_sensitivity, A, B)(x), A, B)[0]

'''
Given epsilon, global_sensitivity, A, B, return the L2 cost for the truncated laplacian mechanism.
'''
def truncated_laplace_L2_eval(epsilon, global_sensitivity, A, B):
    return integrate.quad(lambda x: x**2 * truncated_laplace(epsilon, global_sensitivity, A, B)(x), A, B)[0]

# Example for getting resutls for truncated laplacian
def truncated_laplacian_example(): 
    epsilon = 1e-4 # change here for different epsilon values
    global_sensitivity = 1 # change here for different global sensitivity
    A = -10 # change here for different left truncation values
    a, b = process(epsilon, global_sensitivity, A, -A) # It is necessary to call process function to get the appropriate bounds
    L1_cost = truncated_laplace_L1_eval(epsilon, global_sensitivity, a, b)
    L2_cost = truncated_laplace_L2_eval(epsilon, global_sensitivity, a, b)