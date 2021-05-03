from utils import *
#from plots_2C import *

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
