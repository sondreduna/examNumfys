from utils import *
from plots_2B import *

@nb.njit()
def SDEstep(V,N,dt,beta,tau):

    Psi = 1 - np.exp(-dt * beta * V[1]/N)
    Pir = 1 - np.exp(-dt * 1/tau)
    
    Dsi = np.random.binomial(V[0],Psi)
    Dir = np.random.binomial(V[1],Pir)

    return np.array([V[0] - Dsi,
                     V[1] + Dsi - Dir,
                     V[2] + Dir
                    ])

@nb.njit()
def stochSIR(v_0,tN,dt,beta,tau):
    N = np.sum(v_0)

    T = np.arange(0,tN+dt,dt)
    n = len(T)
    V = np.zeros((n,3))
    V[0,:] = v_0
    for i in range(1,n):
        V[i] = SDEstep(V[i-1],N,dt,beta,tau)

    return T,V

@nb.njit(parallel = True)
def sweep(v_0,tN,dt,beta,tau,batch = 500):

    X = np.zeros(batch)
    
    for i in nb.prange(batch):
        T, v = stochSIR(v_0,tN,dt,beta,tau)

        #if np.max(v[:,1]) <= 2 * v_0[1] and slope(T,v[:,1]) <= 0:
        if log_slope(T,v[:,1]) <= 0:
            X[i] = 0.
        else:
            X[i] = 1.

    return X
        

def outbreak_probability():

    Is = np.arange(1,11,dtype = int)

    N    = 100000
    
    tN   = 30  
    dt   = 0.005
    beta = 0.25
    tau  = 10

    # probability of outbreak for each I
    P    = np.zeros(np.shape(Is))
    stds = np.zeros(np.shape(Is))
    
    for i,I in enumerate(tqdm(Is)):
        v_0  = np.array([N - I,I,0])
        X    = sweep(v_0,tN,dt,beta,tau)
        p    = np.mean(X)
        
        std  = np.sqrt(p * (1 - p)/np.size(X))
        
        P[i]    = p
        stds[i] = std

    np.save(DATA_PATH + "2Bc_P.npy",P)
    np.save(DATA_PATH + "2Bc_std.npy",stds)
    return P, stds
    
if __name__ == "__main__":

    # prob a
    
    N    = 100000
    I    = 10
    
    v_0  = np.array([N - I,I,0])
    tN   = 180 
    dt   = 0.005
    beta = 0.25
    tau  = 10

    V    = np.zeros((10,int(tN/dt)+1,3))
    Ts   = np.arange(0,tN +dt , dt)
    
    for i in range(10):
        _,v  = stochSIR(v_0,tN,dt,beta,tau)
        V[i] = v 

    # normalise V before saving:
    V = V/N

    np.save(DATA_PATH + "2Ba_T.npy",Ts) 
    np.save(DATA_PATH + "2Ba_V.npy",V)

    T = np.load(DATA_PATH + "2Aa_T.npy")
    v = np.load(DATA_PATH + "2Aa_v.npy") 
    
    plot_sir_stoch(T,v,Ts,V)

    # prob b

    plot_infected_stoch(T,v,Ts,V)
    
    # prob c

    p,s = outbreak_probability()
    
    plot_probabilities(p,s,x = np.arange(1,11), path = "2Bc_prob.pdf" )
