from utils import *
from plots_2D import *

beta  = 0.55
ra    = 0.1
rs    = 1.0
fs    = 0.6
fa    = 0.4
tau_E = 3
tau_I = 7

@nb.njit()
def SEIIaR_commuter_step(X,Pse,Pei,Peia,Pir,Piar):
    """
    Function for doing one step of the stochastic SEIIaR commuter model.

    Parameters
    ----------
    X : array
        Previous state of system, i.e. [S,E,I,I_a,R].
    Pse : float
        Probability of transitioning from S to E
    Pei : float
        Probability of transitioning from E to I
    Peia : float
        Probability of transitioning from E to Ia
    Pir  : float
        Probability of transitioning from I to R
    Piar : float
        Probability of transitioning from Ia to R

    Returns
    -------
    X(t + dt ) : array
        Next state of system. 

    """

    Dse          = np.random.binomial(X[0],Pse)
    Dei,Deia,Dee = np.random.multinomial(X[1], (Pei,Peia,1-Pei-Peia) )
    Dir          = np.random.binomial(X[2], Pir)
    Diar         = np.random.binomial(X[3], Piar)
    """
    D = np.array([- Dse,
                  - Dei - Deia + Dse,
                  - Dir + Dei,
                  - Diar+ Deia,
                    Dir + Diar])
    return X + D
    """
    return np.array([X[0] - Dse,
                     X[1] - Dei - Deia + Dse,
                     X[2] - Dir + Dei,
                     X[3] - Diar + Deia,
                     X[4] + Dir + Diar])

    
@nb.njit()
def SEIIaR_commuter(M,X_0,tN,dt):
    """
    Function for solving the time evolution of the stochastic SEIIaR commuter model.
    
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
    X : array
        The state of the system for each time in T.  
    
    """

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

    X          = np.zeros((n,m,m,5),dtype = np.int64)
    X[0,:,:,:] = X_0

    # The loop below assumes that the simulation is
    # runned for a whole number of days, with 0.5 divisible by dt,
    # so that the number of steps are evenly split into night and day.
    
    assert( int(0.5 / dt) * dt  == 0.5 )

    step_length = int(1/(2*dt))
    days       = int(tN)
    
    N_day   = np.sum(M,axis = 0)
    N_night = np.sum(M,axis = 1)
    
    for day in range(days):

        i = day * 2 * step_length # current start index
        
        for j in range(step_length):

            # Night simulation
                    
            I = X[i+j,:,:,2:4]
            I = np.sum(I, axis = 0)            
            Pse = 1 - np.exp(- dt * beta * 1/N_night * ( rs * I[:,0] + ra * I[:,1] ))
            
            for k in range(m):
                # If there are no people in the current town, Pse will be nan,
                # so the value should be the same as the previous value 
                if N_night[k] == 0:
                    X[i+j+1,k,:,:] = X[i+j,k,:,:]
                    continue
                for l in range(m):
                    X[i+j+1,k,l,:] = SEIIaR_commuter_step(X[i+j,k,l,:],Pse[k],Pei,Peia,Pir,Piar)

        i += step_length

        for j in range(step_length):

            # Day simulation 
        
            I = X[i+j,:,:,2:4]
            I = np.sum(I, axis = 1)
            Pse = 1 - np.exp(- dt * beta * 1/N_day * ( rs * I[:,0] + ra * I[:,1] ))
            for k in range(m):
                # If there are no people in the current town, Pse will be nan,
                # so the value should be the same as the previous value 
                if N_day[k] == 0:
                    X[i+j+1,:,k,:] = X[i+j,:,k,:]
                    continue
                for l in range(m):
                    X[i+j+1,l,k,:] = SEIIaR_commuter_step(X[i+j,l,k,:],Pse[k],Pei,Peia,Pir,Piar)

    return T, X
        
if __name__ == "__main__":

    # prob a 
    
    E = 25

    M = np.array([[9000,1000],[200,99800]])
    X_0 = np.tensordot(M,np.array([1,0,0,0,0]),axes = 0)
    X_0[0,0] = np.array([9000-E,E,0,0,0])

    tN = 180
    dt = 0.005
    
    T, X = SEIIaR_commuter(M,X_0,tN,dt)

    np.save(DATA_PATH + "2Da_X.npy", X)
    np.save(DATA_PATH + "2Da_T.npy", T)

    plot_cities(T,X,path = "2Da_commuter.pdf" )

    
