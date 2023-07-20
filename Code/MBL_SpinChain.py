from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from numpy.random import default_rng
from numpy import float64, array
from numpy.linalg import eigvalsh, eigh
from scipy.special import binom
from scipy.sparse.linalg import eigsh

class SpinChain:
    
    def __init__(self,L,N_up,W,Delta,pbc=True,J=[1.0,1.0],seed=None,dtype=float64):
        """
        Creates a Chain object representing a 1d Heisenberg chain.
        Args:
            L: int
                The length of the chain
            N_up: int
                The number of "up" spins. If None, the whole Fock space (ie, all S_z sectors) 
                    is included.
            W: float
                The disorder strength. The (random) on-site potential is sampled from a
                    the uniform distribution [-W/2 , W/2].
            Delta: float
                The interaction strength (e.g. coefficient of the nearest-neighbor S_z-S_z term)
            use_pbc: Bool, optional
                Whether or not to use periodic boundary conditions. Defaults to True
            J: array or float, optional
                The X and Y coupling strengths. Can either be a float (in which case X and Y strengths are
                    are the same and equal to the given value), or a 2-component array-like. Defaults to [1.0,1.0]
            seed: int/unsigned int/etc, optional
                Seed for the random number generator used to generate the on-site potential. Defaults 
                    to None, in which case a seed is randomly generated by the RNG
            dtype: dtype, optional
                Data type to use for internal storage of matrix elements. Defaults to numpy.float64
        """
        self.L = L
        self.W = W
        self.Delta = Delta
        self.J = J
        self.N_up = N_up
        self.pbc = pbc
        if type(J) == float:
            self.J = [J,J]
        else:
            self.J = J

        self._rng = default_rng(seed)
        self.on_site = 2*self.W*self._rng.random(self.L) - self.W

        self.dtype = dtype

        self.basis = spin_basis_1d(L=self.L,Nup=self.N_up)
        
        #Initialize the random on-site potential and nearest-neighbor interaction terms
        self.interactions = [[self.Delta,i,i+1] for i in range(self.L-1)]
        if self.pbc:
            self.interactions.append([self.Delta,self.L-1,0])
       
        self.on_site_pot = [[self.on_site[i] , i] for i in range(self.L)]
        
        self.diags = [["zz" , self.interactions] , ["z" , self.on_site_pot]]
        
        #Initialize the spin flip terms
        self._X = [[self.J[0],i,i+1] for i in range(self.L-1)] 
        self._Y = [[self.J[1],i,i+1] for i in range(self.L-1)]
        if self.pbc:
            self._X.append([self.J[0],L-1,0])
            self._Y.append([self.J[1],L-1,0])

        self.off_diags = [['xx', self._X],['yy',self._Y]]
        
        #Construct the Hamiltonian from the above terms
        self._Ham = hamiltonian(self.diags+self.off_diags,[],dtype=self.dtype,basis=self.basis,check_symm=False,check_herm=False,check_pcon=False)
        self.H = self._Ham.static

        self.energies= None
        self.eigenvectors = None        




    #### ****                                           **** ####
    ####     HAMILTONIAN CREATION/MANIPULATION FUNCTIONS     ####
    #### ****                                           **** ####

    def create_H(self):
        """
        Create the Hamiltonian for this Chain.
        """
        self._Ham = hamiltonian(self.diags+self.off_diags,[],dtype=self.dtype,basis=self.basis,check_symm=False,check_herm=False,check_pcon=False)
        self.H = self._Ham.static


    def make_diag(self):
        """
        Update the diagonal terms of this Hamiltonian, and call create_H() to make the Hamiltonian
            matrix.
        """
        self.interactions = [[self.Delta,i,i+1] for i in range(self.L-1)]
        if self.pbc:
            self.interactions.append([self.Delta,self.L-1,0])
        self.on_site_pot = [[self.on_site[i] , i] for i in range(self.L)]
        self.diags = [["zz" , interactions] , ["z" , on_site_pot]]
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
        if self.N_up is not None:
            return int(binom(self.L,self.N_up))
        else:
            return 1 << L

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
