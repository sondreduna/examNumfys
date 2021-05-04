from utils import *

def add_sol(ax,Ts,vs,index,N,label = ""):
    cm_S = mpl.cm.get_cmap('winter')
    cm_I = mpl.cm.get_cmap('autumn')
    cm_R = mpl.cm.get_cmap('summer')
    Ss,Is,Rs = vs.T

    if label == "":
        ax.plot(Ts,Ss, color = cm_S(index/N), ls = "--")
        ax.plot(Ts,Is, color = cm_I(index/N), ls = "--")
        ax.plot(Ts,Rs, color = cm_R(index/N), ls = "--")
    else:
        ax.plot(Ts,Ss, color = cm_S(index/N), ls = "--",label = label[0])
        ax.plot(Ts,Is, color = cm_I(index/N), ls = "--",label = label[1])
        ax.plot(Ts,Rs, color = cm_R(index/N), ls = "--",label = label[2])

def plot_sir_stoch(T,v,Ts,V):
    S,I,R = v[:,:].T
    
    fig, ax = plt.subplots()
    
    ax.plot(T,S, label ="Susceptible", color = "blue")
    ax.plot(T,I, label = "Infected"   , color = "red")
    ax.plot(T,R, label = "Recovered"  , color = "green")

    N = len(V)
    for i in range(N):
        add_sol(ax,Ts,V[i],i,N)
    
    ax.grid(ls ="--")
    
    ax.set_xlabel(r"$T [\mathrm{days}]$")
    ax.set_ylabel(r"Fraction of population")
    
    plt.legend()
    plt.tight_layout()
    fig.savefig(FIG_PATH + "2Ba_SIR.pdf")

def plot_infected_stoch(T,v,Ts,V):
    S,I,R = v[:,:].T
    
    fig, ax = plt.subplots()
    
    ax.plot(T,I, label = "Infected"   , color = "red")
    ax.plot(T,Iexp(T,I[0],0.25,10), 
            label = r"$I(t) = I(0)\exp{\left( \frac{t}{\tau} (\mathcal{R}_0 - 1)\right)}$", 
            color = "black", 
            ls ="--")
    ax.set_yscale("log")
    ax.grid(ls ="--")

    N = len(V)
    
    
    for i in range(N):
        _,Is,_ = V[i].T
        cm_I = mpl.cm.get_cmap('autumn')
        ax.plot(Ts,Is, color = cm_I(i/N), ls = "--")

    ax.set_yscale("log")
    ax.grid(ls ="--")
    
    ax.set_xlabel(r"$T [\mathrm{days}]$")
    ax.set_ylabel(r"Fraction of population")
    
    plt.legend()
    plt.tight_layout()
    fig.savefig(FIG_PATH + "2Bb_I.pdf")


def plot_probabilities(probs,stds, x, path):

    fig,ax =  plt.subplots()
    plt.plot(x,probs,
             label = r"$P(\mathrm{outbreak} | I)$",
             ls = "",
             marker = "o",
             color = "b")
    ax.errorbar(x,y = probs, yerr = stds, xerr = 0 ,
                color="black",
                ls = "",
                label = "Standard deviation")

    plt.xlabel(r"Initial number of infected people, $I$")
    plt.ylabel(r"Probability")

    plt.legend()
    plt.grid(ls = "--")

    plt.tight_layout()

    fig.savefig(FIG_PATH + path)
