from numpy import array, kron, eye, nonzero


from MBL_Measurement import MBL_Measurement as MM

if __name__ == "__main__":

	from sys import argv

	if len(argv) != 7:
		print("Usage: L W Delta T gamma use_pbc")
		exit(0)

	L = int(argv[1])
	W = float(argv[2])
	Delta = float(argv[3])
	T = float(argv[4])
	gamma = float(argv[5])
	use_pbc = (int(argv[6]) != 0)


	sigma_x = array([[0.0,1.0],[1.0,0.0]])
	sigma_y = array([[0.0,-1.0j],[1.0j,0.0]])
	sigma_z = array([[1.0,0.0],[0.0,-1.0]])

	L1 = kron(sigma_x,eye(1 << (L-1)))
	L2 = kron(sigma_y,eye(1 << (L-1)))
	L3 = kron(sigma_z,eye(1 << (L-1)))

	proj_up = array([[1.0,0.0],[0.0,0.0]])
	proj_down = array([[0.0,0.0],[0.0,1.0]])

	P1 = kron(eye(1 << (L-1)),proj_up)
	P2 = kron(eye(1 << (L-1)),proj_down)

	MM_obj = MM(L,None,W,Delta,T,[gamma,gamma,gamma],[L1,L2,L3],[[P1,P2]],True,use_pbc=use_pbc)

	#eigvals, eigvecs = MM_obj.get_slow_modes_dense(0,num=10)
	eigvals, eigvecs = MM_obj.get_slow_modes_dense(0,num=10)
	print(eigvals)
	#print(eigvecs)
	