import numpy as np
import numba as nb
from scipy.optimize import fsolve
from tqdm import tqdm

DATA_PATH = "../data/"
FIG_PATH  = "../fig/"

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import numpy as np

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
    return s - np.exp(- R_0 * (1- s))

def R(r,R_0):
    return r - 1 + np.exp(- R_0 * r )

def S_inf(beta,tau):
    R_0 = beta*tau
    return fsolve(S,.5,args = R_0)[0]

def R_inf(beta,tau):
    R_0 = beta*tau
    return fsolve(R,.5,args = R_0)[0]

def Iexp(T,I_0,beta,tau):
    return I_0 * np.exp(T/tau * (beta*tau - 1))

# needs to be compiled with numba to use in the sweep-method
@nb.njit()
def slope(x,y):
    return ((x*y).mean() - x.mean()*y.mean()) / ((x**2).mean() - (x.mean())**2)

@nb.njit()
def log_slope(x,y):
    x = x
    y = np.log(y)
    return slope(x,y)

def add_slope_marker(ax,slope,pos):
    slope_marker(pos,(round(slope,3),1),ax = ax)
