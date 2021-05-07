from prob_2D import *
from plots_2E import *

@nb.njit()
def SEIIaR_commuter_greedy(M,X_0,tN,dt):
    """
    Function for solving the time evolution of the stochastic SEIIaR commuter model.
    Greedy version:
        This solver only keeps track of the previous and current state of the system
        as this is only what is needed to step forward, and it saves a lot of memory usage. 
    
        Further, it also calculates N(t) : the number of towns with more than 10 infected
        people at each time step, as this is what we are after in problem 2Eb and 2Ec. 
    
    Parameters
    ----------
    M : array
        Population matrix
    X_0 : array
        Initial state of system
    tN : float
        End time, in days.
    dt : float
        Time step, in days.
    
    Returns
    -------
    T : array
        Time values from 0 to tN spaced by dt.
    infected : array
        The number of towns with more than 10 infected people at each point in time, N(t).
    
    """

    # get number of towns
    m    = np.shape(M)[0] 
    
    # set this to ones initially, but change it 
    # for each step, as it depends on the number of infected.
    Pse  = np.ones(m)

    # Calculate the probabilities that do not change with time
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

    N_day   = np.sum(M,axis = 0)
    N_night = np.sum(M,axis = 1)
    
    for day in range(days):

        i = day * 2 * step_length # current start index
        
        for j in range(step_length):

            # Night simulation
            
            I = X_[:,:,2:4]
            I = np.sum(I, axis = 0)
            Pse = 1 - np.exp(- dt * beta * 1/N_night * ( rs * I[:,0] + ra * I[:,1] ))
            for k in range(m):
                if N_night[k] == 0:
                    X[k,:,:] = X_[k,:,:]
                    continue
                
                for l in range(m):
                    if M[k,l] == 0:
                        # do nothing when there are no people in this group
                        X[k,l,:] = X_[k,l,:]
                    else:
                        X[k,l,:] = SEIIaR_commuter_step(X_[k,l,:],Pse[k],Pei,Peia,Pir,Piar)

            # numba cannot handle multiple axis at once in the np.sum function,
            # therefore, I nest the sums  
            inf  = np.sum(np.sum(X[:,:,2:4],axis = 1),axis = -1)
            mask = inf > 10
            infected[i+j + 1] = np.sum(mask)

            # set previous step to current when incrementing j + i 
            X_ = X
            
        i += step_length

        for j in range(step_length):

            # Day simulation 
            
            I = X_[:,:,2:4]
            I = np.sum(I, axis = 1)

            Pse = 1 - np.exp(- dt * beta * 1/N_day * ( rs * I[:,0] + ra * I[:,1] ))
            for k in range(m):
                if N_day[k] == 0:
                    X[:,k,:] = X_[:,k,:]
                    continue
                
                for l in range(m):
                    if M[l,k] == 0:
                        # do nothing when there are no people in this group 
                        X[l,k,:] = X_[l,k,:]
                    else:
                        X[l,k,:] = SEIIaR_commuter_step(X_[l,k,:],Pse[k],Pei,Peia,Pir,Piar)

            inf  = np.sum(np.sum(X[:,:,2:4],axis = 1),axis = -1)
            mask = inf > 10
            infected[i+j + 1] = np.sum(mask)            

            # set previous step to current when incrementing j + i 
            X_ = X
     
    return T, infected

def generate_pop():
    """
    Function for manually generating and saving the population structure 
    for problem 2Ea.

    """

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
    """
    Function for manipulating the population structure in problem 2Eb 
    to be the one asked for in problem 2Ec. 

    """

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
    """
    Simulate the 10 city problem, and plot the results. 

    """
    M = np.genfromtxt(DATA_PATH + "10_city_pop.csv", delimiter=',')
    
    E = 25
    
    X_0 = np.tensordot(M,np.array([1,0,0,0,0]),axes = 0)

    X_0[1,1] = np.array([9500-E,E,0,0,0])

    tN = 180
    dt = 0.005

    XX = np.zeros((10,int(tN/dt)+1,10,10,5))

    for i in tqdm(range(10)):
        T, X = SEIIaR_commuter(M,X_0,tN,dt)

        # check that no people have left the city!
        assert( np.all(np.sum(X, axis = (2,-1)) == np.sum(M,axis = 1)) )
        
        XX[i] = X

    np.save(DATA_PATH + "2Ea_XX.npy", XX)
    np.save(DATA_PATH + "2Ea_T.npy", T)

    plot_10_cities(T,XX,path = "2Ea_commuter.pdf" )


def big_population_sim(ind, homeoffice = False):
    """
    Simulate the big population structure. 

    Parameters
    ----------
    ind : int 
        Index of simulation : used to be able to run the 10 simulations in
        parallel and save the results to unique files. This index ensures that
    
    homeoffice : bool
        If True : use the population structure for problem 2Ec,
        otherwise, use the population structure for problem 2Eb.

    """
    
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
    # be able to run the simulation in
    # a reasonable amount of time.
    # With these parameters, one run takes approximately 3 minutes. 
    
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
    
    # prob a
    ten_city_sim()

    # provide an index to the function call
    # so that 10 simulations can be run simultaneously,
    # e.g. on linux run :

    # for i in {1..10}; do python prob_2E.py $i &done;
    
    # with either the line below prob b or the one below prob c uncommented.
    
    index = sys.argv[1]
    index = int(index)
    
    # prob b
    # big_population_sim(index)

    # prob c
    # big_population_sim(index, homeoffice = True)

    
    # after having run the ten simulations, separately, run
    #plot_infections("2Eb_T.npy","2Eb_inf","2Eb_N.pdf")
    # and
    #plot_infections("2Eb_T_ho.npy","2Eb_inf_ho","2Ec_N.pdf")

