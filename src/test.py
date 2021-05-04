from prob_2A import *
from prob_2B import *
from prob_2C import *
from prob_2D import *

def test_SEIIaR():

    """
    Function for testing the SEIIaR model against the 
    usual SIR model. This requires to adjust the parameters in prob_2C 
    to:
        beta = 0.25
        ra   = 0 
        rs   = 1
        tau_E= 0 (or very small)
        tau_I= 10.
    
    As it is a bit problematic to compare two stochastic models directly, we
    compare the averages over many runs. Initially I thought of setting the seed
    before running each model, but that won't give the same results any way since 
    we generate more random numbers in the SEIIaR model.

    """

    # SEIIaR
    
    N = 100000
    I = 10

    v_0 = np.array([N-I,0,I,0,0]) 
    tN  = 180
    dt  = 0.001
    rs  = 1.

    V  = np.zeros((int(tN/dt)+1,5))
    Ts = np.arange(0,tN+dt,dt)

    n = 100
    
    for i in tqdm(range(n)):
        T,v = SEIIaR(v_0,tN,dt,rs);
        V  += 1/n * v/N

    # Contract the axis of the infected:

    V_seiiar = np.zeros((int(tN/dt)+1,3))
    V_seiiar[:,0] = V[:,0]
    V_seiiar[:,1] = np.sum(V[:,1:4],axis = 1)
    V_seiiar[:,2] = V[:,4]

    # SIR

    N    = 100000
    I    = 10
    
    v_0  = np.array([N - I,I,0])
    tN   = 180 
    dt   = 0.001
    beta = 0.25
    tau  = 10

    V_sir= np.zeros((int(tN/dt)+1,3))
    Ts   = np.arange(0,tN +dt , dt)
    
    for i in tqdm(range(n)):
        _,v    = stochSIR(v_0,tN,dt,beta,tau)
        V_sir += 1/n * v/N

    np.save(DATA_PATH + "test_Ts.npy",Ts)
    np.save(DATA_PATH + "test_vseiiar.npy",V_seiiar)
    np.save(DATA_PATH + "test_vsir.npy",V_sir)
    
    print( np.max(np.abs(V_sir - V_seiiar)))

def plot_sir_comparison(T,v,Ts,V):
    S,I,R = v[:,:].T
    
    fig, ax = plt.subplots()
    
    ax.plot(T,S, label ="Susceptible", color = "blue")
    ax.plot(T,I, label = "Infected"   , color = "red")
    ax.plot(T,R, label = "Recovered"  , color = "green")

    N = len(V)
    for i in range(N):
        if i == 0:
            add_sol(ax,Ts,V[i],i,N,label = [r"$S_{\mathrm{SIR}}$",r"$I_{\mathrm{SIR}}$",r"$R_{\mathrm{SIR}}$"])
        else:
            add_sol(ax,Ts,V[i],i,N,label = [r"$S_{\mathrm{SEIIaR}}$",r"$I_{\mathrm{SEIIaR}}$",r"$R_{\mathrm{SEIIaR}}$"])
    
    ax.grid(ls ="--")
    
    ax.set_xlabel(r"$T [\mathrm{days}]$")
    ax.set_ylabel(r"Fraction of population")
    
    plt.legend()
    plt.tight_layout()
    fig.savefig(FIG_PATH + "test_comparison.pdf")

def test_commuter():

    # check that the infectious people stay in area 1
    # when no-one travels to work in a different area

    E = 25

    M = np.array([[10000,0],[0,100000]])
    X_0 = np.tensordot(M,np.array([1,0,0,0,0]),axes = 0)
    X_0[0,0] = np.array([9000-E,E,0,0,0])

    tN = 180
    dt = 0.002
    
    T, X = SEIIaR_commuter(M,X_0,tN,dt)

    np.save(DATA_PATH + "test_commuter_X.npy", X)
    np.save(DATA_PATH + "test_commuter_T.npy", T)

    plot_cities(T,X, path = "test_commuter.pdf")

    
if __name__ == "__main__":

    # this only makes sense to test if the parameter values
    # are changed, as described in the report, and in the function.
    # test_SEIIaR()

    Ts = np.load("../data/test_Ts.npy")
    v1 = np.load("../data/test_vseiiar.npy")
    v2 = np.load("../data/test_vsir.npy")

    V = np.zeros((2,np.shape(v1)[0],3))

    V[0] = v1
    V[1] = v2

    T = np.load("../data/2Aa_T.npy")
    v = np.load("../data/2Aa_v.npy")
    
    plot_sir_comparison(T,v,Ts,V)

    # test the commuter model:

    test_commuter()

    
