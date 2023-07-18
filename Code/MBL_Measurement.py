from numpy import eye, kron, zeros, complex64

from itertools import product
from functools import reduce

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
		diss -= (gamma/2)*(kron(L.conj().T @ L,I) + kron(I,L.T @ L.conj()))	# -{L^dag L,p}

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

	def __init__(self,L,N,W,U,T,gammas,Ls,use_pbc=True):
		"""
		Create an MBL_Measurement instance, a class provided for convenient parallelization
			of diagonalizing the measurement evolution operators.

		Args:
			L: int
                The length of the chain
            N: int
                The number of fermions in the chain
            W: float
                The disorder strength. The (random) on-site potential is sampled from a
                    the uniform distribution [-W/2 , W/2].
            U: float
                The interaction strength (e.g. coefficient of the nearest-neighbor density-density term)
            T: float
            	The time between measurements, measured in units of hbar/t (where t is the bare 
            		hopping strength of the chain)
            gammas: array-like
            	An array/list of the system-bath coupling strengths \gamma_i appearing in the Lindblad
            		superoperator
            Ls: array-like
            	An array/list of the Krauss operators L_i appearing in the Lindblad superoperator		
            use_pbc: Bool, optional
                Whether or not to use periodic boundary conditions. Defaults to True

		"""

		self.L = L
		self.N = N
		self.W = W
		self.U = U
		self.T = T 		
		self.gammas = gammas
		self.Ls = Ls
		self.use_pbc = use_pbc
