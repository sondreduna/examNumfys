from plots_2D import *

def plot_10_cities(T,X,path):

    """
    Plotting the seiiar for each city:

    """

    fig, ax = plt.subplots(nrows = 5,ncols = 2, figsize = (18,23), sharex = True)

    v1      = np.sum(X[:,:,:,0,:],axis = 2)
    v2      = np.sum(X[:,:,:,1,:],axis = 2)

    Ss,Es,Is,Ias,Rs = v1[0].T

    # Plot first row manually to get labels on one of the curves
    # for each variable.
    
    ax[0,0].plot(T,Ss, ls = "--",label = r"$S$",color ="blue")
    ax[0,0].plot(T,Es, ls = "--",label = r"$E$",color ="red")
    ax[0,0].plot(T,Is, ls = "--",label = r"$I$",color ="orange")
    ax[0,0].plot(T,Ias, ls = "--",label= r"$I_a$",color ="yellow")
    ax[0,0].plot(T,Rs, ls = "--",label=r"$R$",color ="green")

    for i in range(1,10):
        add_seiiar(ax[0,0],T,v1[i])

    Ss,Es,Is,Ias,Rs = v2[0].T
    
    ax[0,1].plot(T,Ss, ls = "--",label = r"$S$",color ="blue")
    ax[0,1].plot(T,Es, ls = "--",label = r"$E$",color ="red")
    ax[0,1].plot(T,Is, ls = "--",label = r"$I$",color ="orange")
    ax[0,1].plot(T,Ias, ls = "--",label= r"$I_a$",color ="yellow")
    ax[0,1].plot(T,Rs, ls = "--",label=r"$R$",color ="green")

    for i in range(1,10):
        add_seiiar(ax[0,1],T,v2[i])
    
    ax[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=5, fancybox=True, shadow=True)

    ax[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=5, fancybox=True, shadow=True)

    ax[0,0].grid(ls ="--")
    ax[0,1].grid(ls ="--")
    ax[0,0].set_title(r"\textbf{Town 1}")
    ax[0,1].set_title(r"\textbf{Town 2}")
    
    for i in range(1,5):
        for j in range(2):
            V = np.sum(X[:,:,:,i*2 + j,:],axis = 2)
            for k in range(10):
                add_seiiar(ax[i,j],T,V[k])
            ax[i,j].set_title(r"\textbf{Town %i}"%(i*2+j + 1))
            ax[i,j].grid(ls = "--")
                
    
    ax[2,0].set_ylabel(r"Number of people")
    
    ax[4,0].set_xlabel(r"$T [\mathrm{days}]$")
    ax[4,1].set_xlabel(r"$T [\mathrm{days}]$")

    plt.tight_layout()

    fig.savefig(FIG_PATH + path)

def plot_infections(T,array_path, path):

    # first, join all the simulated runs:

    I = np.zeros(10,len(T))

    for i in range(10):
        I[i] = np.load(DATA_PATH + array_path +f"_{i}.npy")
    
    fig = plt.figure()
    plt.plot(T,I[0], label = r"$\mathcal{N}(t)$",color ="blue")
    
    for i in range(1,len(I)):
        plt.plot(T,I[i],color= "blue")
        
    plt.xlabel(r"$t [\mathrm{days}]$")
    plt.ylabel(r"$\mathcal{N}(t)$")

    plt.grid(ls = "--")
    
    plt.tight_layout()
    plt.legend()

    fig.savefig(FIG_PATH + path)
    
    
    
