# potentials.py (Complete Modified Version)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import fsolve
from numba import njit

# Existing two-body and three-body potential functions...
# (e.g., morse, lj, buckingham, poly2, axilrod, poly3)

# Define the MLP architecture (must match the training architecture)
class SimpleModel(nn.Module):
    def __init__(self, input_dim, neuron):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, neuron)
        self.fc2 = nn.Linear(neuron, neuron)
        self.fc3 = nn.Linear(neuron, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_MLP_model(model_path, input_dim, neuron):
    '''
    Load the trained MLP model from the specified path.
    
    Parameters:
    - model_path: str, path to the saved MLP model file.
    - input_dim: int, number of input features.
    - neuron: int, number of neurons in hidden layers.
    
    Returns:
    - model: nn.Module, loaded MLP model.
    '''
    model = SimpleModel(input_dim, neuron)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model

def mlp_potential_function(model, scaler):
    '''
    Create a potential energy function using the loaded MLP model.
    
    Parameters:
    - model: nn.Module, loaded MLP model.
    - scaler: sklearn.preprocessing.MinMaxScaler, fitted scaler for input features.
    
    Returns:
    - V_MLP: callable, function that takes r and returns potential energy.
    '''
    def V_MLP(r):
        '''
        Compute potential energy using MLP.
        
        Parameters:
        - r: float or np.ndarray, bond lengths.
        
        Returns:
        - potential: float or np.ndarray, predicted potential energy.
        '''
        # Ensure r is a numpy array
        r = np.atleast_2d(r)
        
        # Feature processing (must match training)
        # Assuming r has shape (N, 3) and corresponds to [x1, x2, x3]
        processed = process_data_batch(r, l=scaler.scale_[0])  # Adjust 'l' as needed
        inputs = torch.tensor(processed, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(inputs)
            potential = outputs.numpy().flatten()
        return potential
    return V_MLP

@njit
def numerical_derivative(V_func, r, delta=1e-5):
    '''
    Compute the numerical derivative of the potential using central difference.
    
    Parameters:
    - V_func: callable, potential energy function.
    - r: float, bond length.
    - delta: float, small perturbation.
    
    Returns:
    - dV/dr: float, derivative of the potential.
    '''
    return (V_func(r + delta) - V_func(r - delta)) / (2 * delta)

def mlp_derivative_function_numba(V_MLP, delta=1e-5):
    '''
    Create a derivative function for the MLP potential using Numba-accelerated numerical differentiation.
    
    Parameters:
    - V_MLP: callable, potential energy function.
    - delta: float, small perturbation.
    
    Returns:
    - dV_MLP_numba: callable, derivative of the potential.
    '''
    @njit
    def dV_MLP_numba(r):
        return numerical_derivative(V_MLP, r, delta)
    
    return dV_MLP_numba

def process_data_batch(data, l):
    '''
    Process raw coordinates into features for the MLP.
    
    Parameters:
    - data: np.ndarray, raw coordinates [r1, r2, r3].
    - l: float, process parameter (from scaler).
    
    Returns:
    - result: np.ndarray, processed features [x1, x2, x3].
    '''
    result = np.zeros_like(data)
    for i, t in enumerate(data):
        r1, r2, r3 = t[0], t[1], t[2]
        y1 = np.exp(-r1 / l)
        y2 = np.exp(-r2 / l)
        y3 = np.exp(-r3 / l)
        p1 = y2 + y3
        p2 = y2**2 + y3**2
        p3 = y1
        x1 = p1
        x2 = p2**(1 / 2)
        x3 = p3
        result[i] = [x1, x2, x3]
    return result

# Example usage:
# model_path = 'path_to_saved_mlp_model.pth'
# scaler = ...  # Load or define the scaler used during training
# model = load_MLP_model(model_path, input_dim=3, neuron=50)
# V_MLP = mlp_potential_function(model, scaler)
# dV_MLP = mlp_derivative_function_numba(V_MLP)


# Two body potentials
def morse(de = 1., alpha = 1., re = 1.):
    '''
    Return two-body morse potential:
    V(r) = De*(1-exp(-a*(r-re)))^2 - De
    
    Keyword arguments:
    de, float
        dissociation energy
    alpha, float
        alpha = we*sqrt(mu/2De)
    re, float
        equilibrium length
    '''
    V = lambda r: de*(1-np.exp(-alpha*(r-re)))**2 - de
    dV = lambda r: 2*alpha*de*(1-np.exp(-alpha*(r-re)))*np.exp(-alpha*(r-re))
    return V, dV

def lj(m=12,n=6,cm=1,cn=1):
    '''
    Return a general Lennard-Jones potential:
    V(r) = cm/r^m - cn/r^n

    Keyword arguments:
    cn, float
        long-range parameter
    cm, float
        short-range parameter
    '''
    V = lambda r: cm/r**(m)-cn/r**(n)
    dV = lambda r: -m*cm/r**(m+1)+n*cn/r**(n+1)
    return V, dV

def buckingham(a=1., b=1., c6 = 1., max = .1):
    '''Usage:
        V = buckingham(**kwargs)
    Buckingham potentials tend to come back down at low r. 
    We fix this by imposing xmin at the turning point "max."
    Return a one-dimensional Buckingham potential:
    V(r) = a*exp(-b*r) - c6/r^6 for r > r_x

    Keyword arguments:
    a, float
        short-range multiplicative factor
    b, float
        short-range exponential factor
    c6, float
        dispersion coefficient
    max, float
        guess of where the maximum is. At short range, 
        Buckingham potentials can reach a maximum and collapse. 
        Enter your nearest $r$ value to this maximum.

    Outputs:
    Buck, function
        buckingham potential
    dBuck, function
        derivative of buckingham potential
    xi, float
        minimum x-value where Buck is defined
    '''
    Buck = lambda r: a*np.exp(-b*r) - c6/r**6
    dBuck = lambda r: -a*b*np.exp(-b*r) + 6*c6/r**7

    # Find the maximum of potential
    xi = fsolve(dBuck,max)
    
    return Buck, dBuck, xi

def poly2(c0, alpha, b, coeff):
    '''
    Polynomial fit of 2-body ab-initio data (10.1063/1.462163)
    V(r) = c0*e^(-alpha*x)/x + sum(c_i*rho^i), rho = x*e^-(b*x)
    
    Inputs:
    c0, float
    alpha, float
    b, float
    coeff, zipped list of the format [(c_i,i)] 
        Keep track of coeffiecients (c_i) and degree (i)
    '''
    v_long = lambda x: sum([i*((x*np.exp(-b*x))**j) for i, j in coeff])
    v_short = lambda x: c0*np.exp(-alpha*x)/x 
    V = lambda x: v_long(x) + v_short(x)
    dv_long = lambda x: sum([i*(-j*(b*x-1)*x**(j-1)*np.exp(-j*b*x)) for i, j in coeff])
    dv_short = lambda x: -c0*(1+alpha*x)*(np.exp(-alpha*x)/x**2)
    dV = lambda x: dv_long(x) + dv_short(x)
    return V, dV

# Three body potentials
def axilrod(C = 0):
    '''
    Return Axilrod-Teller potential
    
    C = V*alpha1*alpha2*alpha3

    V - Ionization energy 
    alpha - atomic polarizability


    '''
    V = lambda r12,r23,r31: C*(1/(r12*r23*r31)**3 - 3*((r12**2-r23**2-r31**2)*
                                                        (r23**2-r31**2-r12**2)*
                                                        (r31**2-r12**2-r23**2))/
                                                    8/(r12*r23*r31)**5)
    
    dvdR12 = lambda r12,r23,r31: -3*C*(r12**6 + r12**4*(r23**2 + r31**2) - 
            5*(r23**2 - r31**2)**2*(r23**2 + r31**2) + 
            r12**2*(3*r23**4 + 2*r23**2*r31**2 + 3*r31**4))/(8*r12**6*r23**5*r31**5)
    
    dvdR23= lambda r12,r23,r31: -3*C*(r23**6 + r23**4*(r31**2 + r12**2) - 
            5*(r31**2 - r12**2)**2*(r31**2 + r12**2) + 
            r23**2*(3*r31**4 + 2*r31**2*r12**2 + 3*r12**4))/(8*r23**6*r31**5*r12**5)
    
    dvdR31= lambda r12,r23,r31: -3*C*(r31**6 + r31**4*(r12**2 + r23**2) - 
            5*(r12**2 - r23**2)**2*(r12**2 + r23**2) + 
            r31**2*(3*r12**4 + 2*r12**2*r23**2 + 3*r23**4))/(8*r31**6*r12**5*r23**5)
    return V, dvdR12, dvdR23, dvdR31

def poly3(b_12, b_23, b_31, coeffs):
    '''
    Polynomial fit of 3-body ab-initio data (10.1063/1.462163)
    sum_{ijk}^{M} (d_ijk*p_12^i*p_23^j*p_31^k), p_ab = x_ab*e^(-b*x_ab)

    ENSURE i+j+k!=i!=j!= k AND i+j+k<=M
    Inputs:
    b12, b23, b31, float
        exponential parameters of p_ab
    coeffs, zipped list of the format [(d_ijk,[i,j,k])] 
        Keep track of coeffiecients and degree
    '''
    v = lambda r12,r23,r31: sum([i*(r12*np.exp(-b_12*r12))**j[0]*(r23*np.exp(-b_23*r23))**
                                 j[1]*(r31*np.exp(-b_31*r31))**j[2] for i,j in coeffs])
    dvdr12 = lambda r12,r23,r31: sum([-j[0]*(b_12*r12-1)/r12*i*(r12*np.exp(-b_12*r12))**
                                      j[0]*(r23*np.exp(-b_23*r23))**
                                      j[1]*(r31*np.exp(-b_31*r31))**j[2] for i, j in coeffs])
    dvdr23 = lambda r12,r23,r31: sum([-j[1]*(b_23*r23-1)/r23*i*(r12*np.exp(-b_12*r12))**
                                      j[0]*(r23*np.exp(-b_23*r23))**
                                      j[1]*(r31*np.exp(-b_31*r31))**j[2] for i, j in coeffs])
    dvdr31 = lambda r12,r23,r31: sum([-j[2]*(b_31*r31-1)/r31*i*(r12*np.exp(-b_12*r12))**
                                      j[0]*(r23*np.exp(-b_23*r23))**
                                      j[1]*(r31*np.exp(-b_31*r31))**j[2] for i, j in coeffs])
    return v, dvdr12, dvdr23, dvdr31