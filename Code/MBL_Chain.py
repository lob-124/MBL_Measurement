from quspin.basis import spinless_fermion_basis_1d
from quspin.operators import hamiltonian
from numpy.random import default_rng
from numpy import float64, array
from numpy.linalg import eigvalsh, eigh
from scipy.special import binom
from scipy.sparse.linalg import eigsh

class Chain:
    
    def __init__(self,L,N,W,U,pbc=True,t=1,seed=None,dtype=float64):
        """
        Creates a Chain object representing a 1d chain with interactions.
        Args:
            L: int
                The length of the chain
            N: int
                The number of fermions in the chain. If None, the whole Fock space (ie, all N sectors) 
                    is included.
            W: float
                The disorder strength. The (random) on-site potential is sampled from a
                    the uniform distribution [-W/2 , W/2].
            U: float
                The interaction strength (e.g. coefficient of the nearest-neighbor density-density term)
            use_pbc: Bool, optional
                Whether or not to use periodic boundary conditions. Defaults to True
            t: float, optional
                The hopping amplitude. Defaults to 1
            seed: int/unsigned int/etc, optional
                Seed for the random number generator used to generate the on-site potential. Defaults 
                    to None, in which case a seed is randomly generated by the RNG
            dtype: dtype, optional
                Data type to use for internal storage of matrix elements. Defaults to numpy.float64
        """
        self.L = L
        self.N = N
        self.W = W
        self.U = U
        self.t = t
        self.pbc = pbc

        self._rng = default_rng(seed)
        self.on_site = 2*self.W*self._rng.random(self.L) - self.W

        self.dtype = dtype

        self.basis = spinless_fermion_basis_1d(L=self.L, Nf = self.N)
        
        #Initialize the random on-site potential and nearest-neighbor interaction terms
        self.interactions = [[self.U,i,i+1] for i in range(self.L-1)]
        if self.pbc:
            self.interactions.append([self.U,self.L-1,0])
       
        self.on_site_pot = [[self.on_site[i] , i] for i in range(self.L)]
        
        self.diags = [["nn" , self.interactions] , ["n" , self.on_site_pot]]
        
        #Initialize the hopping terms
        self.hopping_L = [[self.t,i,i+1] for i in range(self.L-1)] 
        self.hopping_R = [[self.t,i+1, i] for i in range(self.L-1)]
        if self.pbc:
            self.hopping_L.append([self.t,L-1,0])
            self.hopping_R.append([self.t,0,L-1])

        self.hoppings = [['+-', self.hopping_L],['+-',self.hopping_R]]
        
        #Construct the Hamiltonian from the above terms
        self._Ham = hamiltonian(self.diags+self.hoppings,[],dtype=self.dtype,basis=self.basis)
        self.H = self._Ham.static

        self.energies= None
        self.eigenvectors = None        




    #### ****                                           **** ####
    ####     HAMILTONIAN CREATION/MANIPULATION FUNCTIONS     ####
    #### ****                                           **** ####

    def create_H(self):
        """
        Initialize the Hamiltonian for this Chain.
        """
        self._Ham = hamiltonian(self.diags+self.hoppings,[],dtype=self.dtype,basis=self.basis)
        self.H = self._Ham.static


    def make_diag(self):
        self.interactions = [[self.U,i,i+1] for i in range(self.L-1)]
        if self.pbc:
            self.interactions.append([self.U,self.L-1,0])
        self.on_site_pot = [[self.on_site[i] , i] for i in range(self.L)]
        self.diags = [["nn" , interactions] , ["n" , on_site_pot]]
        self.create_H()


    def new_disorder(self,disorder=None):
        """
        Generates a new disorder realization, or sets it to a user-supplied realization, and updates 
            the Hamiltonian.

        Args:
            disorder: array, optional
                The new disorder realization. If None, then generates a new one automatically. Defaults 
                    to None
        """
        if disorder is not None:
            self.on_site = disorder
        else:
            self.on_site = 2*self.W*self._rng.random(self.L) - self.W
        self.make_diag()





    #### ****                                     **** ####
    ####     HAMILTONIAN DIAGONALIZATION FUNCTIONS     ####
    #### ****                                     **** ####

    def spectrum(self):
        """
        Compute and return the spectrum of eigenvalues for the current Hamiltonian.
        """
        return eigvalsh(self.H.todense())


    def all_states(self):
        """
        Compute and return the spectrum of eigenvalues - and their associated eigenvectors - for
            the current Hamiltonian.
        """
        return eigh(self.H.todense())


    def spectrum_extrema(self,k,mode='LM',sigma=None):
        """
        Use sparse diagonalization to return the extrema of the spectrum for the current Hamiltonian.
        """
        return eigsh(self.H,k=k,sigma=sigma,which=mode,return_eigenvectors=False)


    def states_extrema(self,k,mode='LM',sigma=None):
        """
        Use sparse diagonalization to return the extrema of the spectrum for the current Hamiltonian.
        Returns the eigenvectors as well
        """
        return eigsh(self.H,k=k,sigma=sigma,which=mode,return_eigenvectors=True)




    #### ****              **** ####
    ####     MISC FUNCTIONS     ####
    #### ****              **** ####

    def dimension(self):
        """
        Return the size of the Hamiltonian matrix for this chain.
        """
        if self.N is not None:
            return int(binom(self.L,self.N))
        else:
            return 1 << self.L


    def get_H(self,dense=False):
        """
        Return the Hamiltonian matrix.

        Args:
            dense: Bool, optional
                Whether to return the dense (True) or sparse (False) version of the 
                    Hamiltonian matrix. Defaults to False

        Returns:
            H: array or scipy sparse matrix
                The Hamiltonian matrix
        """
        if dense:
            return self.H.todense()
        else:
            return self.H
