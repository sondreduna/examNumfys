from plots_2C import *

def plot_cities(T,X, path):

    """
    Plotting the seiiar for each city:

    """

    fig, ax = plt.subplots(nrows = 2, figsize = (13,13), sharex = True)

    v1      = np.sum(X[:,0,:,:],axis = 1)
    v2      = np.sum(X[:,1,:,:],axis = 1)

    Ss,Es,Is,Ias,Rs = v1.T

    # Plot first row manually to get labels on one of the curves
    # for each variable.
    
    ax[0].plot(T,Ss, ls = "--",label = r"$S$",color ="blue")
    ax[0].plot(T,Es, ls = "--",label = r"$E$",color ="red")
    ax[0].plot(T,Is, ls = "--",label = r"$I$",color ="orange")
    ax[0].plot(T,Ias, ls = "--",label= r"$I_a$",color ="yellow")
    ax[0].plot(T,Rs, ls = "--",label=r"$R$",color ="green")

    # this neat feature is borrowed from:
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=5, fancybox=True, shadow=True)
    
    add_seiiar(ax[1],T,v2)

    ax[0].grid(ls ="--")
    ax[0].set_ylabel(r"Number of people")
    ax[1].grid(ls ="--")
    ax[1].set_xlabel(r"$t [\mathrm{days}]$")
    ax[1].set_ylabel(r"Number of people")

    plt.tight_layout()

    fig.savefig(FIG_PATH + path)
