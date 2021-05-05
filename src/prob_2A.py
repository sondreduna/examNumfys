from ode import *
from plots_2A import *

beta = 0.25
tau  = 10

# If using nb.njit on this, remember to recompile the function
# each different run when changing beta or tau, as they are
# treated as constants if not. 

# @nb.njit()
def sir_rhs(t,v):
    return np.array([- beta * v[0] * v[1],
                     beta * v[0] *v[1] - v[1]/tau,
                     v[1]/tau
    ])


class SIRSolver(ODESolver):
    """
    Simple class for solving the sir equations, built on the same
    principles as that of solving the LLG equation in exercise 2.

    """
    
    def __init__(self,t0,y0,tN,h,beta_,tau_,method = "RK4"):
        super().__init__(sir_rhs,t0,y0,tN,h,method)

        global beta
        global tau

        beta = beta_
        tau  = tau_


def find_max_beta(beta_0,tol = 1e-6):
    """
    Simple approach to finding the largest beta for which 
    I is smaller than 0.2.

    """

    beta = beta_0

    I = 1e-4
    R = 0
    S = 1 - I - R

    v_0 = np.array([S,I,R])
    
    TN  = 180
    dt  = 0.05
    
    sir  = SIRSolver(0,v_0,TN,dt,beta_0,10)
    _,v  = sir()

    Imax = np.max(v[:,1])
    Ilim = 0.2

    err = np.abs(Imax - Ilim)
    
    while err > tol:
        if Ilim > Imax :
            beta = beta * 2**(err)
        else:
            beta = beta * 2**(-err)

        sir  = SIRSolver(0,v_0,TN,dt,beta,10)
        _,v  = sir()

        err = np.abs(np.max(v[:,1]) - Ilim)

    return beta, err
        

def vaccination_test(R_0):
    """
    Increasing the ratio of vaccinated people until the 
    log-slope of the infected is sufficiently far from that of the
    exponential growth in the early stage approximation 
    """

    I = 1e-4
    R = R_0
    S = 1 - I - R

    v_0 = np.array([S,I,R])
    
    TN  = 50
    dt  = 0.05
    
    sir  = SIRSolver(0,v_0,TN,dt,0.25,10)
    T,v  = sir()

    slope_ref = beta - 1/tau 

    # avoid using t = 0 as it is problematic when
    # computing the log. Also v[0] = 0 occasionally.
    
    slope     = log_slope(T[1:300],v[1:300,1]) 
    
    while slope > 0:
        R = R * 2 ** ( slope)
        S = 1 - I - R
        v_0 = np.array([S,I,R])
        
        sir  = SIRSolver(0,v_0,TN,dt,beta,10)
        T,v  = sir()
        
        slope     = log_slope(T[1:300],v[1:300,1])

    return R, slope
    

if __name__ == "__main__":

    ## prob. a
    I = 1e-4
    R = 0
    S = 1 - I - R

    v_0 = np.array([S,I,R])
    TN  = 180
    dt  = 0.005

    sir = SIRSolver(0,v_0,TN,dt,0.25,10)
    T,v = sir()

    np.save(DATA_PATH + "2Aa_v.npy",v)
    np.save(DATA_PATH + "2Aa_T.npy",T)

    S_ = S_inf(beta,tau)
    R_ = R_inf(beta,tau)
    
    plot_sir(T,v,S_,R_)

    ## prob b
    
    plot_infected(T,v)

    ## prob c
    
    beta_max, Imax = find_max_beta(0.20)
    np.savetxt(DATA_PATH + "betamax.txt", np.array([beta_max,Imax]))
    
    ## prob d

    Rmin, slopemin = vaccination_test(0.1)
    np.savetxt(DATA_PATH + "Rmin.txt", np.array([Rmin,slopemin]))

    

