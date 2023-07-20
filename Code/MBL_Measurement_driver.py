from numpy import array, kron, eye, ceil, real, imag

from MBL_Measurement import MBL_Measurement as MM

from multiprocessing import Pool
from struct import pack

if __name__ == "__main__":

	from sys import argv

	if len(argv) != 10:
		print("Usage: L W Delta T gamma use_pbc trials num_threads outfile")
		exit(0)

	L = int(argv[1])
	W = float(argv[2])
	Delta = float(argv[3])
	T = float(argv[4])
	gamma = float(argv[5])
	use_pbc = (int(argv[6]) != 0)
	trials = int(argv[7])
	num_threads = int(argv[8])
	outfile = argv[9]


	sigma_x = array([[0.0,1.0],[1.0,0.0]])
	sigma_y = array([[0.0,-1.0j],[1.0j,0.0]])
	sigma_z = array([[1.0,0.0],[0.0,-1.0]])

	L1 = kron(sigma_x,eye(1 << (L-1)))
	L2 = kron(sigma_y,eye(1 << (L-1)))
	L3 = kron(sigma_z,eye(1 << (L-1)))

	proj_up = array([[1.0,0.0],[0.0,0.0]])
	proj_down = array([[0.0,0.0],[0.0,1.0]])

	if L % 2:
		#Odd system size: measure on middle site
		P1 = kron(kron(eye(1 << (L-1)//2),proj_up),eye(1 << (L-1)//2))
		P2 = kron(kron(eye(1 << (L-1)//2),proj_down),eye(1 << (L-1)//2))
	else:
		#Even system size: measure at site L/2+1 (with site indexing starting from 1)
		P1 = kron(kron(eye(1 << L//2),proj_up),eye(1 << (L//2-1)))
		P2 = kron(kron(eye(1 << L//2),proj_down),eye(1 << (L//2-1)))
		

	MM_obj = MM(L,None,W,Delta,T,[gamma,gamma,gamma],[L1,L2,L3],[[P1,P2]],True,use_pbc=use_pbc)

	#Find the steady states and slow modes
	evals = []
	steady_states = []
	slow_modes = []
	with Pool(processes=num_threads) as p:
		for eigenvalues , eigenvectors in p.map(MM_obj.get_slow_modes_dense,range(trials),chunksize=int(ceil(trials/num_threads))):
			evals.append(eigenvalues[-2])
			print(eigenvalues[-2])
			steady_states.append(eigenvectors[-1])
			slow_modes.append(eigenvectors[-2])


	#Save the data
	with open(outfile,"wb") as f:
		f.write(pack("i",trials))

		for _eval in evals:
			f.write(pack("dd",real(_eval),imag(_eval)))

		for _evec in steady_states:
			for elem in _evec:
				f.write(pack("dd",real(elem),imag(elem)))

		for _evec in slow_modes:
			for elem in _evec:
				f.write(pack("dd",real(elem),imag(elem)))

