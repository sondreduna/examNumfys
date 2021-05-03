from prob_2C import *

beta  = 0.55
ra    = 0.1
rs    = 1.
fs    = 0.6
fa    = 0.4
tau_E = 3
tau_I = 7

r     = np.array([rs,ra]) 


@nb.njit()
def SEIIaR_commuter_step(X,Pse,Pei,Peia,Pir,Piar):

    Dse          = np.random.binomial(X[0],Pse)
    Dei,Deia,Dee = np.random.multinomial(X[1], (Pei,Peia,1-Pei-Peia) )
    Dir          = np.random.binomial(X[2], Pir)
    Diar         = np.random.binomial(X[3], Piar)

    return np.array([X[0] - Dse,
                     X[1] - Dei - Deia + Dse,
                     X[2] - Dir + Dei,
                     X[3] - Diar + Deia,
                     X[4] + Dir + Diar])

@nb.njit()
def SEIIaR_commuter(M,X_0,tN,dt):
    
    # set this to ones initially, but change it 
    # for each step, as it depends on the number of infected.

    m    = np.shape(M)[0] 
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
    
    for day in range(days):

        i = day * 2 * step_length # current start index
        
        for j in range(step_length):
            
            N = np.sum(M,axis = 0)
            I = X[i+j,:,:,2:4]
            I = np.sum(I, axis = 0)
            Pse = 1 - np.exp(- dt * beta * 1/N * ( rs * I[:,0] + ra * I[:,1] ))
            for k in range(m):
                for l in range(m):
                    X[i+j+1,k,l,:] = SEIIaR_commuter_step(X[i+j,k,l,:],Pse[k],Pei,Peia,Pir,Piar)

        i += step_length

        for j in range(step_length):
            
            N = np.sum(M,axis = 1)
            I = X[i+j,:,:,2:4]
            I = np.sum(I, axis = 1)
            Pse = 1 - np.exp(- dt * beta * 1/N * ( rs * I[:,0] + ra * I[:,1] ))
            for k in range(m):
                for l in range(m):
                    X[i+j+1,l,k,:] = SEIIaR_commuter_step(X[i+j,l,k,:],Pse[k],Pei,Peia,Pir,Piar)

    return T, X
        
        
    
    
