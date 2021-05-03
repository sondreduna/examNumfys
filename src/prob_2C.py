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

    Pse  = 1 - np.exp(-dt * beta * 1/N *(rs * V[2] + ra * V[3]))
    Pei  = fs * (1 - np.exp(-dt/tau_E))
    Peia = fa * (1 - np.exp(-dt/tau_E))
    Pir  = 1 - np.exp(-dt/tau_I)
    Piar = 1 - np.exp(-dt/tau_I)

    Dse          = np.random.binomial(V[0],Pse)
    Dei,Deia,Dee = np.random.multinomial(int(V[1]), (Pei,Peia,1-Pei-Peia) )
    Dir          = np.random.binomial(V[2], Pir)
    Diar         = np.random.binomial(V[3], Piar)

    return np.array([V[0] - Dse,
                     V[1] - Dei - Deia + Dse,
                     V[2] - Dir + Dei,
                     V[3] - Diar + Deia,
                     V[4] + Dir + Diar])

@nb.njit()
def SEIIaR(v_0,tN,dt,rs):
    N = np.sum(v_0)

    T = np.arange(0,tN+dt,dt)
    n = len(T)
    V = np.zeros((n,5))
    V[0,:] = v_0
    for i in range(1,n):
        V[i] = SEIIaRstep(V[i-1],N,dt,rs)

    return T,V

@nb.njit(parallel = True)
def sweep_SEIIaR(v_0,tN,dt,rs,batch = 500):

    X = np.zeros(batch)
    
    for i in nb.prange(batch):
        T, v = SEIIaR(v_0,tN,dt,rs)

        # use only the initial stage of the simulation to calculate
        # the slope, to avoid nans due to the logarithms
        
        if slope(T[:],np.sum(v[:,1:3], axis = 1 )) <= 0:
            X[i] = 0.
        else:
            X[i] = 1.

    return X

def self_isolation(num_vals = 100):

    N = 100000
    E = 25

    v_0 = np.array([N-E,E,0,0,0]) 
    tN  = 20
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
    for i in range(10):
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

    p,s,r = self_isolation()

    plot_probabilities_rs(p,s,x = r,path = "2Cb_probs.pdf")
