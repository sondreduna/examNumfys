from utils import *
from plots_2C import *

beta  = 0.55
ra    = 0.1
fs    = 0.6
fa    = 0.4
tau_E = 3
tau_I = 7

@nb.njit()
def SEIIaRstep(V,N,dt,rs):
    """
    Function for doing one step of the stochastic SEIIaR model.

    Note that the implementation only directly allows for changing 
    one parameter of the model, namley rs. This is done to save computation time
    as only this parameter is changed in the problems.

    Parameters
    ----------
    V : array
        Previous state of system, i.e. [S,E,I,I_a,R].
    N : int
        Number of people in population.
    dt : float
        Time step.
    rs : float
        r_s parameter of the model. 

    Returns
    -------
    V(t + dt ) : array
        Next state of system. 

    """

    Pse  = 1 - np.exp(-dt * beta * 1/N *(rs * V[2] + ra * V[3]))
    Pei  = fs * (1 - np.exp(-dt/tau_E))
    Peia = fa * (1 - np.exp(-dt/tau_E))
    Pir  = 1 - np.exp(-dt/tau_I)
    Piar = 1 - np.exp(-dt/tau_I)

    Dse          = np.random.binomial(V[0],Pse)
    Dei,Deia,Dee = np.random.multinomial(V[1], (Pei,Peia,1-Pei-Peia) )
    Dir          = np.random.binomial(V[2], Pir)
    Diar         = np.random.binomial(V[3], Piar)

    D = np.array([-Dse,-Dei-Deia+Dse,-Dir+Dei, -Diar+Deia,Dir+Diar])
    return V + D
    """
    return np.array([V[0] - Dse,
                     V[1] - Dei - Deia + Dse,
                     V[2] - Dir + Dei,
                     V[3] - Diar + Deia,
                     V[4] + Dir + Diar])
    """
@nb.njit()
def SEIIaR(v_0,tN,dt,rs):
    """
    Function for solving the time evolution of the stochastic SEIIaR model.
    
    Parameters
    ----------
    v_0 : array
        Initial state of system
    tN : float
        End time, in days.
    dt : float
        Time step, in days.
    rs : float
        r_s parameter of the model. 
    
    Returns
    -------
    T : array
        Time values from 0 to tN spaced by dt.
    V : array
        The state of the system for each time in T.  
    
    """
    
    N = np.sum(v_0)

    T = np.arange(0,tN+dt,dt)
    n = len(T)
    V = np.zeros((n,5), dtype = np.int64)
    V[0,:] = v_0
    for i in range(1,n):
        V[i] = SEIIaRstep(V[i-1],N,dt,rs)

    return T,V

@nb.njit(parallel = True)
def sweep_SEIIaR(v_0,tN,dt,rs,batch = 500):
    """
    Function for simulating a batch of (default) 500 
    runs of the stochastic SIR model, and classifying 
    each of the simulations as yielding an exponentially growing 
    or decreasing outbreak, based on the slope of the number of infected people.

    Parameters
    ----------
    v_0 : array
        Initial state of system
    tN : float
        End time, in days.
    dt : float
        Time step, in days.
    rs : float
        r_s parameter of the model. 
    batch : int
        Number of runs to do.

    Returns
    -------
    X : array
        Array of 1s or 0s, corresponding to exponential outbreak or not respectively,
        for the batch number of runs.

    """

    X = np.zeros(batch)
    
    for i in nb.prange(batch):
        T, v = SEIIaR(v_0,tN,dt,rs)

        # use only the initial stage of the simulation to calculate
        # the slope, to avoid nans due to the logarithms
        
        if log_slope(T[:],np.sum(v[:,1:4], axis = 1 )) <= 0:
            X[i] = 0.
        else:
            X[i] = 1.

    return X

def self_isolation(num_vals = 100):
    """
    Function for finding the probabilty of an outbreak 
    as a function of the self-isolation-rate, rs. 
    
    This is described more thoroughly in the report. 
    
    Parameters
    ----------
    num_vals : int 
        Number of r_s values to test
    
    Returns
    -------
    P : array
        probability of an outbreak for all rs values
    stds : array
        The associated standard deviation for each of these estimates.
    Rs   : array
        The r_s values considered. 

    """

    N = 100000
    E = 25

    v_0 = np.array([N-E,E,0,0,0]) 
    tN  = 30
    dt  = 0.005

    V  = np.zeros((10,int(tN/dt)+1,5))
    Ts = np.arange(0,tN+dt,dt)

    # probability of outbreak for each value of rs

    P    = np.zeros(num_vals)
    stds = np.zeros(num_vals)
    Rs   = np.linspace(1e-3,1,num_vals)

    for i in tqdm(range(num_vals)):

        X       = sweep_SEIIaR(v_0,tN,dt,Rs[i])
        p       = np.mean(X)
        P[i]    = p
        stds[i] = np.sqrt(p * (1- p)/np.size(X))

    np.save(DATA_PATH + "2Cb_P.npy",P)
    np.save(DATA_PATH + "2Cb_std.npy",stds)
    np.save(DATA_PATH + "2Cb_Rs.npy",Rs)
    return P, stds, Rs


if __name__ == "__main__":

    # prob a
    
    N = 100000
    E = 25

    v_0 = np.array([N-E,E,0,0,0]) 
    tN  = 180
    dt  = 0.005
    rs  = 1.

    V  = np.zeros((10,int(tN/dt)+1,5))
    Ts = np.arange(0,tN+dt,dt)
    for i in tqdm(range(10)):
        T,v = SEIIaR(v_0,tN,dt,rs);
        V[i] = v

    # Normalise V before saving

    V = V/N
    np.save(DATA_PATH + "2Ca_T.npy",Ts) 
    np.save(DATA_PATH + "2Ca_V.npy",V)

    T = np.load(DATA_PATH + "2Aa_T.npy")
    v = np.load(DATA_PATH + "2Aa_v.npy") 

    # add E I and I_a to compare with the analytical model:
    V_ = np.zeros((10,int(tN/dt)+1,3))
    V_[:,:,0] = V[:,:,0]
    V_[:,:,1] = V[:,:,1] + V[:,:,2] + V[:,:,3]
    V_[:,:,2] = V[:,:,4]
    
    plot_sir_comp(T,v,Ts,V_)
    plot_seiiar(Ts,V)

    # prob b

    self_isolation()
    # run self_isolation() with r_a = 1
    # and add a _a to all the filenames to make
    # the second array used in the plot below
    
    plot_probabilities_rs(path = "2Cb_probs.pdf")
