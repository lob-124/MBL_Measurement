from numpy import eye, kron, zeros, complex64
from numpy.linalg import eig,eigh,eigvalsh
from scipy.linalg import expm

from itertools import product
from functools import reduce

from time import perf_counter

from MBL_SpinChain import SpinChain

def commutator(H):
	"""
	Given a matrix H, computes the matrix of the commutator superoperator [H,-].
	"""
	dim = H.shape[0]
	I = eye(dim)

	return kron(H,I) - kron(I,H.T)


def Lindbladian(gammas,Ls):
	"""
	Given a list of Krauss operators L_i and coupling strengths \gamma_i, computes the matrix of
		the dissipation superoperator
						\mathcal{L}[p] = \sum_i \gamma_i[L_i p L_i^\dag - {L_i^\dag L_i,p}/2]
	"""
	dim = Ls[0].shape[0]
	I = eye(dim)

	diss = zeros((dim**2,dim**2),dtype=complex64)
	for gamma,L in zip(gammas,Ls):
		diss += gamma*kron(L,L.conj())		#LpL^dag
		diss -= (gamma/2)*(kron(L.conj().T @ L,I) + kron(I,L.T @ L.conj()))	# -{L^dag L,p}/2

	return diss


def measurements(projectors):
	"""
	Given a list of measurements and the projectors to all possible outcomes, computes the
		superoperator associated with performing the measurements.

	Args:
		projectors: array-like
			A nested list/array, whose first index corresponds to the measurements being performed, and
				whose second index enumerates the projectors onto all possible outcomes of each measurement

	Returns:
		superop: array
			The superoperator for all the projective measurements, as a matrix
	"""
	dim = projectors[0][0].shape[0]
	superop = zeros((dim**2,dim**2))
	
	#Each projector P acts as p -> PpP, so we need to select one projector from each measurement 
	#	and apply them consecutively in this way. We then loop over all possible selections 
	for tup in product(*projectors):
		left_op = reduce(lambda a,b: a @ b, tup)
		superop += kron(left_op,left_op.T)

	return superop




class MBL_Measurement:

	def __init__(self,L,N_up,W,Delta,T,gammas,Ls,projectors,is_hermitian,use_pbc=True):
		"""
		Create an MBL_Measurement instance, a class provided for convenient parallelization
			of diagonalizing the measurement evolution operators.

		Args:
			L: int
                The length of the chain
            N: int
                The number of fermions in the chain. If None, the whole Fock space is used
            W: float
                The disorder strength. The (random) on-site potential is sampled from a
                    the uniform distribution [-W/2 , W/2].
            Delta: float
                The interaction strength (e.g. coefficient of the nearest-neighbor S_z-S_z term)
            T: float
            	The time between measurements, measured in units of hbar/t (where t is the bare 
            		hopping strength of the chain)
            gammas: array-like
            	An array/list of the system-bath coupling strengths \gamma_i appearing in the Lindblad
            		superoperator
            Ls: array-like
            	An array/list of the Krauss operators L_i appearing in the Lindblad superoperator		
            projectors: array-like
				A nested list/array, whose first index corresponds to the measurements being performed, and
					whose second index enumerates the projectors onto all possible outcomes of each measurement
            is_hermitian: Bool
            	Whether or not the jump operators are all hermitian. If they are, then the Lindbladian
            		superoperator is as well
            use_pbc: Bool, optional
                Whether or not to use periodic boundary conditions. Defaults to True

		"""
		self.L = L
		self.N_up = N_up
		self.W = W
		self.Delta = Delta
		self.T = T 	
		self.is_hermitian = is_hermitian
		self.use_pbc = use_pbc

		#The Lindblad dissipator superoperator
		self.diss = Lindbladian(gammas,Ls)

		#The superoperator for the projective measurements
		self.proj = measurements(projectors)


	def get_slow_modes_dense(self,seed,num=2):
		"""
		Return the slowest modes (i.e., the eigenstates of the evolution operator with the largest 
			(by magnitude) eigenvalues) of the evolution operator for one period:
							U = P exp(LT)
			where P,L are the projection and Lindblad superoperators. 

		This method uses DENSE matrices for both construction and diagonalization of the superoperators

		Args:
			seed: int/unsigned int.etc
				Seed for the rng used to degenerate the disorder in the system Hamiltonian.
			num: int, optional
				Number of modes to return. Defaults to two, corresponding to the steady state and the
					slowest mode.
		"""

		chain = SpinChain(self.L,self.N_up,self.W,self.Delta,pbc=self.use_pbc,seed=seed)

		H = chain.get_H(dense=True)

		comm = commutator(H)
		Lind = -1j*comm + self.diss

		evo = self.proj @ expm(Lind*self.T)

		
		# if self.is_hermitian:
		# 	eigenvalues , eigenvectors = eigh(evo)
		# 	return eigenvalues[-num:] , eigenvectors[:,-num:]
		# else:
		eigenvalues , eigenvectors = eig(evo)
		indices = abs(eigenvalues).argsort()
		return eigenvalues[indices[-num:]] , eigenvectors[:,indices[-num:]]


	def get_slow_modes_sparse(self,seed,num=2):
		"""
		Return the slowest modes (i.e., the eigenstates of the evolution operator with the largest 
			(by magnitude) eigenvalues) of the evolution operator for one period:
							U = P exp(LT)
			where P,L are the projection and Lindblad superoperators. 

		This method uses SPARSE matrices for both construction and diagonalization of the superoperators

		Args:
			seed: int/unsigned int.etc
				Seed for the rng used to degenerate the disorder in the system Hamiltonian.
			num: int, optional
				Number of modes to return. Defaults to two, corresponding to the steady state and the
					slowest mode.
		"""

		return None
		#To come!
		
		# chain = Chain(self.L,self.N,self.W,self.U,pbc=self.use_pbc,seed=seed)

		# H = chain.get_H(dense=False)

		# comm = commutator(H)
		# L = -1j*comm + diss

		# evo = proj @ expm(L*self.T)

		# if self.is_hermitian:
		# 	eigenvalues , eigenvectors = eigh(evo)
		# 	return eigenvalues[-num:] , eigenvectors[:,-num:]
		# else:
		# 	eigenvalues , eigenvectors = eig(evo)
		# 	indices = abs(eigenvalues).argsort()
		# 	return eigenvalues[indices[-num:]] , eigenvectors[:,indices[-num:]]




