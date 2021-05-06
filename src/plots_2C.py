from utils import *
from plots_2B import *

def plot_sir_comp(T,v,Ts,V):
    S,I,R = v[:,:].T
    
    fig, ax = plt.subplots()
    
    ax.plot(T,S, label ="Susceptible", color = "blue")
    ax.plot(T,I, label = "Infected"   , color = "red")
    ax.plot(T,R, label = "Recovered"  , color = "green")

    N = len(V)
    for i in range(N):
        add_sol(ax,Ts,V[i],i,N)
    
    ax.grid(ls ="--")
    
    ax.set_xlabel(r"$t [\mathrm{days}]$")
    ax.set_ylabel(r"Fraction of population")
    
    plt.legend()
    plt.tight_layout()

    fig.savefig(FIG_PATH + "2Ca_comp.pdf")


def add_seiiar(ax,Ts,vs):

    Ss,Es,Is,Ias,Rs = vs.T
    ax.plot(Ts,Ss, ls = "--",color ="blue")
    ax.plot(Ts,Es, ls = "--",color ="red")
    ax.plot(Ts,Is, ls = "--",color ="orange")
    ax.plot(Ts,Ias, ls = "--",color ="yellow")
    ax.plot(Ts,Rs, ls = "--",color ="green")

def plot_seiiar(Ts,V):
    fig, ax = plt.subplots()

    # plot the deterministic solution as well:

    T = np.load(DATA_PATH + "2Aa_T.npy")
    v = np.load(DATA_PATH + "2Aa_v.npy")

    S,I,R = v[:,:].T

    ax.plot(T,S, color = "blue")
    ax.plot(T,I, color = "red")
    ax.plot(T,R, color = "green")
    
    N = len(V)

    Ss,Es,Is,Ias,Rs = V[0,:,:].T

    # Plot first row manually to get labels on one of the curves
    # for each variable.
    
    ax.plot(Ts,Ss, ls = "--",label = r"$S$",color ="blue")
    ax.plot(Ts,Es, ls = "--",label = r"$E$",color ="red")
    ax.plot(Ts,Is, ls = "--",label = r"$I$",color ="orange")
    ax.plot(Ts,Ias, ls = "--",label= r"$I_a$",color ="yellow")
    ax.plot(Ts,Rs, ls = "--",label=r"$R$",color ="green")
    
    for i in range(1,N):
        add_seiiar(ax,Ts,V[i])
    
    ax.grid(ls ="--")
    
    ax.set_xlabel(r"$t [\mathrm{days}]$")
    ax.set_ylabel(r"Fraction of population")
    
    plt.legend()
    plt.tight_layout()
    fig.savefig(FIG_PATH + "2Ca_SEIIaR.pdf")

    
def plot_probabilities_rs(path):

    probs = np.load(DATA_PATH + "2Cb_P.npy")
    stds  = np.load(DATA_PATH + "2Cb_std.npy")
    x     = np.load(DATA_PATH + "2Cb_Rs.npy")
    
    fig,ax =  plt.subplots()
    plt.plot(x,probs,
             label = r"$P(\mathrm{outbreak} | r_s)$",
             ls = "--",
             marker = ".",
             color = "b")
    ax.errorbar(x,y = probs, yerr = stds, xerr = 0 ,
                color="black",
                ls = "",
                label = "Standard deviation")

    # add same results but with r_a = 1
    
    probs = np.load(DATA_PATH + "2Cb_P_a.npy")
    stds  = np.load(DATA_PATH + "2Cb_std_a.npy")
    x     = np.load(DATA_PATH + "2Cb_Rs_a.npy")

    plt.plot(x,probs,
             label = r"$P(\mathrm{outbreak} | r_s)$, $r_a = 1$",
             ls = "--",
             marker = ".",
             color = "r")
    ax.errorbar(x,y = probs, yerr = stds, xerr = 0 ,
                color="black",
                ls = "",
                label = "Standard deviation")

    plt.xlabel(r"Infectiousness when symptomatic, $r_s$")
    plt.ylabel(r"Probability")

    plt.legend()
    plt.grid(ls = "--")

    plt.tight_layout()

    fig.savefig(FIG_PATH + path)
