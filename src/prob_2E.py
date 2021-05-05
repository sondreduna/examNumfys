from prob_2D import *
from plots_2E import *

@nb.njit()
def SEIIaR_commuter_greedy(M,X_0,tN,dt):

    m    = np.shape(M)[0] 
    
    # set this to ones initially, but change it 
    # for each step, as it depends on the number of infected.

    Pse  = np.ones(m)

    Pei  = fs * (1 - np.exp(-dt/tau_E))
    Peia = fa * (1 - np.exp(-dt/tau_E))
    Pir  = 1 - np.exp(-dt/tau_I)
    Piar = 1 - np.exp(-dt/tau_I)

    T = np.arange(0,tN+dt,dt)
    n = len(T)
    
    # store only the previous and current time step
    X_         = np.zeros((m,m,5),dtype = np.int64)
    X          = np.zeros((m,m,5),dtype = np.int64)

    X_         = X_0 

    # The loop below assumes that the simulation is
    # runned for a whole number of days, with 0.5 divisible by dt,
    # so that the number of steps are evenly split into night and day.
    
    assert( int(0.5 / dt) * dt  == 0.5 )

    step_length = int(1/(2*dt))
    days        = int(tN)

    # We only use this function to do problems 2Eb and c,
    # so we are only interested in the number of municipalities with
    # more than 10 infected for each time step:

    infected   = np.zeros(n) 
    
    for day in range(days):

        i = day * 2 * step_length # current start index
        
        for j in range(step_length):

            # Daytime simulation
            
            N = np.sum(M,axis = 0)
            I = X_[:,:,2:4]
            I = np.sum(I, axis = 0)
            Pse = 1 - np.exp(- dt * beta * 1/N * ( rs * I[:,0] + ra * I[:,1] ))
            for k in range(m):
                for l in range(m):        
                    X[k,l,:] = SEIIaR_commuter_step(X_[k,l,:],Pse[k],Pei,Peia,Pir,Piar)

            # numba cannot handle multiple axis at once in the sum function,
            # therefore, I nest the sums rather. 
            inf  = np.sum(np.sum(X[:,:,2:4],axis = 1),axis = -1)
            mask = inf > 10
            infected[i+j + 1] = np.sum(mask)
            
            X_ = X
        i += step_length

        for j in range(step_length):

            # Night simulation 
            
            N = np.sum(M,axis = 1)
            I = X_[:,:,2:4]
            I = np.sum(I, axis = 1)

            Pse = 1 - np.exp(- dt * beta * 1/N * ( rs * I[:,0] + ra * I[:,1] ))
            for k in range(m):
                for l in range(m):        
                    X[l,k,:] = SEIIaR_commuter_step(X_[l,k,:],Pse[k],Pei,Peia,Pir,Piar)

            inf  = np.sum(np.sum(X[:,:,2:4],axis = 1),axis = -1)
            mask = inf > 10
            infected[i+j + 1] = np.sum(mask)            
        
            X_ = X
     
    return T, infected

def generate_pop():

    M = np.zeros((10,10),dtype = np.int64)

    M[0] = np.array([198600,100,100,100,100,1000,0,0,0,0])
    M[1] = np.array([500,9500,0,0,0,0,0,0,0,0])
    M[2] = np.array([500,0,9500,0,0,0,0,0,0,0])
    M[3] = np.array([500,0,0,9500,0,0,0,0,0,0])
    M[4] = np.array([500,0,0,0,9500,0,0,0,0,0])
    M[5] = np.array([1000,0,0,0,0,498200,200,200,200,200])
    M[6] = np.array([0,0,0,0,0,1000,19000,0,0,0])
    M[7] = np.array([0,0,0,0,0,1000,0,19000,0,0])
    M[8] = np.array([0,0,0,0,0,1000,0,0,19000,0])
    M[9] = np.array([0,0,0,0,0,1000,0,0,0,19000])

    np.savetxt(DATA_PATH + "10_city_pop.csv",M, delimiter = ",")

def work_from_home():

    M = np.genfromtxt(DATA_PATH + "population_structure.csv", delimiter=',')
    M = M.astype("int64")

    for i in range(356):
        mask = np.arange(355)
        mask[i:] += 1
        Ns   = np.sum(M[i,:])
        M[i,mask] = np.around(M[i,mask]/10)
        M[i,i] = Ns - np.sum(M[i,mask])

    np.savetxt(DATA_PATH + "population_structure_ho.csv",M,delimiter=",")
    
def ten_city_sim():
    M = np.genfromtxt(DATA_PATH + "10_city_pop.csv", delimiter=',')
    
    E = 25
    
    X_0 = np.tensordot(M,np.array([1,0,0,0,0]),axes = 0)

    X_0[1,1] = np.array([9500-E,E,0,0,0])

    tN = 180
    dt = 0.005

    XX = np.zeros((10,int(tN/dt)+1,10,10,5))

    for i in tqdm(range(10)):
        T, X = SEIIaR_commuter(M,X_0,tN,dt)
        XX[i] = X

    np.save(DATA_PATH + "2Ea_XX.npy", XX)
    np.save(DATA_PATH + "2Ea_T.npy", T)

    plot_10_cities(T,XX,path = "2Ea_commuter.pdf" )

@nb.njit(parallel = True)
def big_sweep(M,X_0,tN,dt):

    infected = np.zeros((10,int(tN/dt)+1))
    T = np.arange(0,tN + dt, dt)
    for i in nb.prange(10):
        _, I = SEIIaR_commuter_greedy(M,X_0,tN,dt)
        infected[i] = I
        
    return T, infected

def big_population_sim(ind, homeoffice = False):
    if homeoffice:
        M = np.genfromtxt(DATA_PATH + "population_structure_ho.csv", delimiter=',')
    else:
        M = np.genfromtxt(DATA_PATH + "population_structure.csv", delimiter=',')
    
    E = 50
    
    X_0 = np.tensordot(M,np.array([1,0,0,0,0]),axes = 0)
    N_0 = M[0,0]
    X_0[0,0] = np.array([N_0-E,E,0,0,0])

    # To draw numbers from the discrete distributions, the array values
    # have to be integers.
    
    X_0 = X_0.astype("int64")
    M   = M.astype("int64")
    
    # choose a moderately high dt to
    # be even able to run the simulation
    
    tN = 180
    dt = 0.01  
        
    T, I = SEIIaR_commuter_greedy(M,X_0,tN,dt)

    if homeoffice:
        np.save(DATA_PATH + f"2Eb_inf_ho_{ind}.npy", I)
        np.save(DATA_PATH + "2Eb_T_ho.npy", T)
    else:
        np.save(DATA_PATH + f"2Eb_inf_{ind}.npy", I)
        np.save(DATA_PATH + "2Eb_T.npy", T)


import sys

if __name__ == "__main__":

    # provide an index to the function call
    # so that 10 runs can be run simultaneously
    
    index = sys.argv[1]
    index = int(index)
    # prob a
    # ten_city_sim()
    
    # prob b
    # big_population_sim(index)

    # prob c
    # big_population_sim(index, homeoffice = True)

    
    # after having run this, separately, run
    # plot_infections("2Eb_T.npy","2Eb_inf","2Eb_N.pdf")
    # and
    plot_infections("2Eb_T_ho.npy","2Eb_inf_ho","2Ec_N.pdf")

