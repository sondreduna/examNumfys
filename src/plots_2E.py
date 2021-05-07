from plots_2D import *

def plot_10_cities(T,X,path):

    """
    Plotting the seiiar for each city:

    """

    fig, ax = plt.subplots(nrows = 5,ncols = 2, figsize = (18,23), sharex = True)

    v1      = np.sum(X[:,:,0,:,:],axis = 2)
    v2      = np.sum(X[:,:,1,:,:],axis = 2)

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
            V = np.sum(X[:,:,i*2 + j,:,:],axis = 2)
            for k in range(10):
                add_seiiar(ax[i,j],T,V[k])
            ax[i,j].set_title(r"\textbf{Town %i}"%(i*2+j + 1))
            ax[i,j].grid(ls = "--")
                
    
    ax[2,0].set_ylabel(r"Number of people")
    
    ax[4,0].set_xlabel(r"$t [\mathrm{days}]$")
    ax[4,1].set_xlabel(r"$t [\mathrm{days}]$")

    plt.tight_layout()

    fig.savefig(FIG_PATH + path)

def plot_infections(T_path,array_path, path):

    T = np.load(DATA_PATH + T_path)
    
    # first, join all the simulated runs:

    I = np.zeros((10,len(T)))

    for i in range(1,11):
        I[i-1] = np.load(DATA_PATH + array_path +f"_{i}.npy")
    
    fig = plt.figure()
    
    
    for i in range(len(I)):
        plt.plot(T,I[i],color= "blue",alpha = 0.5)

    plt.plot(T,np.mean(I,axis=0), label = r"$\overline{\mathcal{N}}(t)$",color ="black")
        
    plt.xlabel(r"$t [\mathrm{days}]$")
    plt.ylabel(r"$\mathcal{N}(t)$")

    plt.grid(ls = "--")
    
    plt.tight_layout()
    plt.legend()

    fig.savefig(FIG_PATH + path)
    
def plot_matrices():

    pop  = np.genfromtxt('../data/population_structure.csv', delimiter=',')
    pop2 = np.genfromtxt('../data/population_structure_ho.csv', delimiter=',')

    fig, ax = plt.subplots(ncols = 2, figsize = (14,7))

    ax[0].set_title(r"\textbf{Normal population structure}")
    ax[1].set_title(r"\textbf{Population structure, $90\%$ home office}")

    # add small number to each entry to get finite values
    # for the entries being 0 
    ax[0].imshow(np.log(pop + 1e-16))
    ax[1].imshow(np.log(pop2 + 1e-16))

    ax[0].axis("scaled")
    ax[1].axis("scaled")

    plt.tight_layout()
    fig.savefig("../fig/matrices.pdf")
    
