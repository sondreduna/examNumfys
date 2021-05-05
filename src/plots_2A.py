from utils import * 

def plot_sir(T,v,S_,R_):
    S,I,R = v[:,:].T
    
    fig, ax = plt.subplots()
    
    ax.plot(T,S, label ="Susceptible", color = "blue")
    ax.plot(T,I, label = "Infected"   , color = "red")
    ax.plot(T,R, label = "Recovered"  , color = "green")
    
    ax.plot(T,np.ones_like(T)*S_, color= "blue", label =r"$S(\infty)$",ls="--")
    ax.plot(T,np.ones_like(T)*R_, color= "green", label =r"$R(\infty)$",ls="--")
    
    ax.grid(ls ="--")
    
    ax.set_xlabel(r"$t [\mathrm{days}]$")
    ax.set_ylabel(r"Fraction of population")
    
    plt.legend()
    plt.tight_layout()
    fig.savefig(FIG_PATH + "2Aa_SIR.pdf")

def plot_infected(T,v):
    S,I,R = v[:,:].T
    
    fig, ax = plt.subplots()
    
    ax.plot(T,I, label = "Infected"   , color = "red")
    ax.plot(T,Iexp(T,I[0],0.25,10), 
            label = r"$I(t) = I(0)\exp{\left( \frac{t}{\tau} (\mathcal{R}_0 - 1)\right)}$", 
            color = "black", 
            ls ="--")
    
    ax.set_yscale("log")
    ax.grid(ls ="--")
    
    ax.set_xlabel(r"$t [\mathrm{days}]$")
    ax.set_ylabel(r"Fraction of population")
    
    plt.legend()
    plt.tight_layout()
    
    fig.savefig(FIG_PATH + "2Ab_I.pdf")
