import numpy as np
import numba as nb
from scipy.optimize import fsolve
from tqdm import tqdm
from time import time

DATA_PATH = "../data/"
FIG_PATH  = "../fig/"

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl

from scipy.stats import linregress
from mpltools.annotation import slope_marker

fontsize = 24
newparams = {'axes.titlesize': fontsize,
             'axes.labelsize': fontsize,
             'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 
             'legend.fontsize': fontsize,
             'figure.titlesize': fontsize,
             'legend.handlelength': 1.5, 
             'lines.linewidth': 2,
             'lines.markersize': 7,
             'figure.figsize': (11, 7), 
             'figure.dpi':200,
             'text.usetex' : True,
             'font.family' : "sans-serif"
            }

plt.rcParams.update(newparams)


def S(s,R_0):
    """
    Function defining S(\infty) = 0
    """
    return s - np.exp(- R_0 * (1- s))

def R(r,R_0):
    """
    Function defining R(\infty) = 0
    """
    return r - 1 + np.exp(- R_0 * r )

def S_inf(beta,tau):
    """
    Function to find S(\infty)
    """
    R_0 = beta*tau
    return fsolve(S,.5,args = R_0)[0]

def R_inf(beta,tau):
    """
    Function to find R(\infty)
    """
    R_0 = beta*tau
    return fsolve(R,.5,args = R_0)[0]

def Iexp(T,I_0,beta,tau):
    """
    Analytical expression for the early stages of the pandemic,
    as described in the report and problem sheet. 

    """
    return I_0 * np.exp(T/tau * (beta*tau - 1))

# needs to be compiled with numba to use in the sweep-method
@nb.njit()
def slope(x,y):
    """
    Calculate the slope of the graph defined by the set (x,y)

    """
    return ((x*y).mean() - x.mean()*y.mean()) / ((x**2).mean() - (x.mean())**2)

@nb.njit()
def log_slope(x,y):
    """
    Calculate the slope of the graph defined by the set (x,y),
    with logarithmic scales on the y-axis.
    """
    x = x
    y = np.log(y)

    # Ignore the values at which y is infinite, i.e. occuring when
    # we take np.log(0), which happens quite often (!)
    
    mask = np.isfinite(y)
    
    return slope(x[mask],y[mask])

def add_slope_marker(ax,slope,pos):
    slope_marker(pos,(round(slope,3),1),ax = ax)
