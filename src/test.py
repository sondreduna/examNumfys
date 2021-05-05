from prob_2A import *
from prob_2B import *
from prob_2C import *
from prob_2D import *
from prob_2E import *

def test_timesteps():
    """
    Simple test to figure out what timesteps are 
    feasible for the stochastic model. 

    """

    beta = 0.25
    tau  = 10
    
    # --- initial value for deterministic model ---
    I = 1e-4
    R = 0
    S = 1 - I - R

    v_0 = np.array([S,I,R])
    # --- initial value for stochastic model    ---
    
    N    = 100000
    I    = 10
    
    v_0_  = np.array([N - I,I,0])

    # ---------------------------------------------
    
    tN  = 250
    dts = np.logspace(-3,0.5,10)

    # --- Reference solution at t = 300 ---

    S_ = S_inf(beta,tau)
    R_ = R_inf(beta,tau)

    # ------------------------------------

    errs = np.zeros((2,2,10))
    batch = 100

    for i,dt in enumerate(tqdm(dts)):
        sir = SIRSolver(0,v_0,tN,dt,0.25,10)
        T,v = sir()

        errs[0,0,i] = np.abs(v[-1,0] - S_)
        errs[1,0,i] = np.abs(v[-1,2] - R_)

        T  = np.arange(0,tN + dt, dt)
        v_ = np.zeros((len(T),3))
        for j in range(batch):
            _, V = stochSIR(v_0_,tN,dt,beta,tau)
            v_ += 1/batch * V
            
        v_ = v_/N

        errs[0,1,i] = np.abs(v_[-1,0] - S_)
        errs[1,1,i] = np.abs(v_[-1,2] - R_)


    np.save(DATA_PATH + "dts.npy", dts)
    np.save(DATA_PATH + "errs.npy", errs)

def test_timesteps_deterministic():
    """
    Simple test for determining which time steps are
    feasible for the deterministic model.

    """

    beta = 0.25
    tau  = 10
    
    # --- initial value for deterministic model ---
    I = 1e-4
    R = 0
    S = 1 - I - R

    v_0 = np.array([S,I,R])
    
    tN  = 50
    dts = np.logspace(-3,0.5,10)

    # --- Reference solution at t = 300 ---

    dt_ref = 1e-4
    
    sir = SIRSolver(0,v_0,tN,dt_ref,0.25,10)
    T,v = sir(True)
    S_ = v[-1,0]

    # ------------------------------------

    errs = np.zeros((2,10))

    for i,dt in enumerate(tqdm(dts)):
        sir = SIRSolver(0,v_0,tN,dt,0.25,10)
        tic = time()
        T,v = sir()
        toc = time()
        
        errs[0,i] = np.abs(v[-1,0] - S_)
        errs[1,i] = toc - tic 

    np.save(DATA_PATH + "dts_det.npy", dts)
    np.save(DATA_PATH + "errs_det.npy", errs)

def plot_timestep():

    dts = np.load(DATA_PATH + "dts.npy")
    errs = np.load(DATA_PATH + "errs.npy")

    fig = plt.figure()

    plt.scatter(dts, errs[0,0,:],
            marker = "1",
            color = "blue",
            label = r"$|S(250) - S(\infty)|$ Deterministic")
    plt.scatter(dts, errs[1,0,:],
            marker = "2",
            color = "red", 
            label = r"$|R(250) - R(\infty)|$ Deterministic")
    plt.scatter(dts,
            errs[0,1,:],
            marker = "1",color = "green", 
            label = r"$|S(250) - S(\infty)|$ Stochastic")
    plt.scatter(dts, 
            errs[1,1,:],
            marker = "2",
            color = "yellow",
            label = r"$|R(250) -R(\infty)|$ Stochastic")

    plt.xscale("log")
    plt.yscale("log")

    plt.grid(ls = "--")
    plt.legend()
    plt.tight_layout()

    fig.savefig(FIG_PATH + "timestep.pdf")
    
def plot_timestep_deterministic():

    dts = np.load(DATA_PATH + "dts_det.npy")
    errs = np.load(DATA_PATH + "errs_det.npy")

    fig, ax = plt.subplots()

    ax.scatter(dts, errs[0,:],
            marker = "1",
            color = "blue",
            label = r"\texttt{error} Deterministic")

    ax.legend()

    ax2 = ax.twinx()
    ax2.scatter(dts, errs[1,:],
            marker = "1",
            color = "red")

    ax.set_ylabel("Global error",color ="blue")
    
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylabel("Runtime [seconds]",color ="red")

    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.grid(ls = "--")
    plt.tight_layout()

    fig.savefig(FIG_PATH + "timestep_det.pdf")

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
    before running each model, but that won't give the same results anyway since 
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
    
    ax.set_xlabel(r"$t [\mathrm{days}]$")
    ax.set_ylabel(r"Fraction of population")
    
    plt.legend()
    plt.tight_layout()
    fig.savefig(FIG_PATH + "test_comparison.pdf")

def test_commuter():
    """
    Simple function for testing that the commuter model
    behaves roughly as we expect.  

    """

    # check that the infectious people stay in area 1
    # when no-one travels to work in a different area

    E = 25

    M = np.array([[10000,0],[0,100000]])
    X_0 = np.tensordot(M,np.array([1,0,0,0,0]),axes = 0)
    X_0[0,0] = np.array([10000-E,E,0,0,0])

    tN = 180
    dt = 0.002
    
    T, X = SEIIaR_commuter(M,X_0,tN,dt)

    np.save(DATA_PATH + "test_commuter_X.npy", X)
    np.save(DATA_PATH + "test_commuter_T.npy", T)

    plot_cities(T,X, path = "test_commuter.pdf")

    # check that the distributions are roughly in synch
    # when all of those in town 1 travel to work in town 2

    E = 25

    M = np.array([[1,9999],[0,100000]])
    X_0 = np.tensordot(M,np.array([1,0,0,0,0]),axes = 0)
    X_0[0,1] = np.array([9999-E,E,0,0,0])

    tN = 180
    dt = 0.002
    
    T, X = SEIIaR_commuter(M,X_0,tN,dt)

    np.save(DATA_PATH + "test_commuter_X2.npy", X)
    np.save(DATA_PATH + "test_commuter_T2.npy", T)

    plot_cities(T,X, path = "test_commuter_2.pdf")

def test_greedy_commuter():
    """
    Function for testing the greedy commuter function one the smaller 
    population, to see that it work as expected. 

    """
    
    M = np.genfromtxt(DATA_PATH + "10_city_pop.csv", delimiter=',')
    
    E = 25
    
    X_0 = np.tensordot(M,np.array([1,0,0,0,0]),axes = 0)
    X_0[1,1] = np.array([9500-E,E,0,0,0])

    X_0 = X_0.astype("int64")

    tN = 180
    dt = 0.02

    for i in tqdm(range(10)):
        T, I = SEIIaR_commuter_greedy(M,X_0,tN,dt)


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

    # test greedy commuter:

    # test_greedy_commuter()
