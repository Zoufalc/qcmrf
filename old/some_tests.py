import sys
import numpy as np
from scipy.linalg import expm
np.set_printoptions(threshold=sys.maxsize,linewidth=1024)

import itertools
from colored import fg, bg, attr

from qiskit.opflow import I, X, Z, Plus, Minus, H, Zero, One
from qiskit.compiler import transpile
from qiskit import QuantumRegister, QuantumCircuit
from qiskit import Aer, assemble

####################################### input structure

C = [[0]] # clique structure, 4-var circle
#C = [[0,1],[1,2],[2,3],[0,3]] # clique structure, 4-var circle

n = len(np.unique(np.array(C).flatten())) # number of (qu)bits
d = 0
for c in C:
	m = len(c)
	d = d + (2**m)
	
dim = 2**n

####################################### utility funcs

def grouped(iterable, n):
	return zip(*[iter(iterable)]*n)
	
def merge(L):
	R = []
	for (M0,M1) in grouped(L,2):
		R.append((((I+Z)/2)^M0) + (((I-Z)/2)^M1))
	return R
	
def merge_all(L):
	R = L
	while len(R)>1:
		R = merge(R)
	return R[0]
	
turquoise = '#179c7d'
orange = '#eb6a0a'
blue = '#1f82c0'
	
def format_head(h,t):
	return fg(turquoise)+attr('bold')+h+attr('reset')+': '+fg('white')+attr('bold')+t+attr('reset')
	
def format_val(val):
	if type(val) is int:
		return fg('white')+attr('bold')+str(val)+attr('reset')
	else:
		return fg('white')+attr('bold')+np.format_float_scientific(val, precision=4)+attr('reset')
		
def format_math(math):
	brackets  = ['(',')']
	operators = ['+','-','*','/','_','^']
	mcol = turquoise
	bcol = 'white'
	ocol = orange
	for b in brackets:
		math = math.replace(b,fg(bcol)+b+fg(mcol))
	for o in operators:
		math = math.replace(o,fg(ocol)+o+fg(mcol))
	return fg(mcol)+math+attr('reset')

####################################### model funcs

def genPhi(c,y):
	result = 1
	
	plus  = [v for i,v in enumerate(c) if not y[i]] # 0
	minus = [v for i,v in enumerate(c) if     y[i]] # 1

	for i in range(n):
		f = I
		if i in minus:
			f = (I-Z)/2
		elif i in plus:
			f = (I+Z)/2

		result = result^f
		
	return result

def genHamiltonian():
	L = []
	R = 0
	i = 0
	for l,c in enumerate(C):
		Ly = []
		for r,y in enumerate(list(itertools.product([0, 1], repeat=len(c)))):
			Phi = genPhi(c,y)
			theta = -0.1*(i+1) #np.random.uniform() # we need a negative MRF
			R += Phi * -theta
			Ly.append((theta,Phi)) # list of all factors that belong to same clique
			i = i + 1
		L.append(Ly) # list of factor-lists, one factor-list per clique

	return R,L

def genPhaseFactors(ey):
	t = np.sqrt((-1-ey)/(-1+ey))
	
	phi1 = np.angle( ( (1+t*1j) + ey*(1-t*1j) ) / 2 )
	phi2 = np.real(np.angle( np.sqrt(0.5) * np.sqrt(1 - ey) * (t + 1j) ))
	
	return np.array([phi1, phi2])
	
def genUphi(U,phi):
	RZ1 = (phi[0] * Z).exp_i() ^ (I^n) # ignored sign of phi
	RZ2 = (phi[1] * Z).exp_i() ^ (I^n) # ignored sign of phi
	return RZ2 @ U @ RZ2 @ U # ignored conjugate transpose of first U

# works only if Eigenvalues of A are bounded by 1	
def uniEmbedding(A):
	return (X^((I^n)-A)) + (Z^A)

# returns unitary if A is unitary	
def conjugateBlocks(A):
	return (((I+Z)/2)^A) + (((I-Z)/2)^(~A))

# works only if number of cliques is a power of two
def expH_from_list_blocked(beta, L0, lnZ=0):
	R = []
	for Ly in L0:
		L = [genUphi(uniEmbedding(Phi0), genPhaseFactors(np.exp(beta*(w0 - (lnZ/len(C)))))) for w0,Phi0 in Ly]
		R.append(merge_all(L))
	M = merge_all(R)
	O = H^(I^int(n+1+np.log2(d)))
	return O @ conjugateBlocks(M) @ (~O)
	
def expH_from_list_real_algebraic(beta, L0, lnZ=0):
	RESULT = I^(n+1)
	for L in L0:
		for (w0,Phi0) in L:
			U = genUphi(uniEmbedding(Phi0), genPhaseFactors(np.exp(beta*(w0 - (lnZ/len(C))))))
			RESULT = (U+(~U))/2 @ RESULT # non-unitary extraction of real part
	return RESULT
	
def expH_from_list_unreal(beta, L0, lnZ=0):
	RESULT = I^(n+1)
	for L in L0:
		for (w0,Phi0) in L:
			U = genUphi(uniEmbedding(Phi0), genPhaseFactors(np.exp(beta*(w0 - (lnZ/len(C))))))
			RESULT = U @ RESULT # unitary extraction of real part
	return RESULT

def expH_from_list_real_RUS(beta, L0, lnZ=0):
	qr = QuantumRegister(n+1+d, 'q') # one ancille per factor
	circ = QuantumCircuit(n+1+d,n+1+d)
	for i in range(n+1):
		circ.h(qr[i])
	
	i = 0 # enumerate 0..d-1
	for ii,L in enumerate(L0):
		for jj,(w0,Phi0) in enumerate(L):
			w = 0.5*w0 - (lnZ/len(C))
			assert w < 0
			
			U = genUphi(uniEmbedding(Phi0), genPhaseFactors(np.exp(beta*w)))
			
			u = conjugateBlocks(U).to_circuit().to_instruction(label='U_C'+str(ii)+'_y'+str(jj))
			
			circ.h(qr[n+1+i])
			circ.append(u, [qr[j] for j in range(n+1)]+[qr[n+1+i]])
			circ.h(qr[n+1+i])
			
			i = i + 1
			
	circ.measure([qr[j] for j in range(n+1+d)],range(n+1+d))
	return circ

#######################################

HAM,L  = genHamiltonian() # L is list of factors
beta = 1

R0   = expm(-beta*HAM.to_matrix()) # exp(-βH) via numpy for debugging

#######################################

print("NODES ("+format_math('n')+"):",n," PARAMETERS ("+format_math('d')+"):",d)

#######################################

if False:
	print()
	print(format_head("CHECK 3","expH_from_list_blocked(..);"))
	print("  * This shows that the new block factorization with ("+format_math('1+log_2(d)'))
	print("    extra ancillas) has correct values of each factor on the diagonal.")

	R3   = expH_from_list_blocked(beta, L)
	r3   = R3.to_matrix()

	r = np.eye(2*dim) # 2*dim = 2**(n+1)
	for i in range(d):
		a = (i)   * (2*dim)
		b = (i+1) * (2*dim)
		B = r3[a:b,a:b]
		r = B @ r

	print("  * "+format_math('l2')+"-Error between",format_math('exp(-βH)'),"and the product over")
	print("    first",format_math('d'),"diagonal blocks of block factorization:", format_val(np.linalg.norm(R0 - r[:dim,:dim], ord=2)))

#######################################
print()
print(format_head("BUILD CIRCUIT","expH_from_list_real_RUS(..);"))
print("  * Eq. (3) with RUL")

R2b   = expH_from_list_real_RUS(beta, L)

OL = 3
print()
print(format_head("TRANSPILATION","transpile(..)"))
print("  * For {cx, id, rz, sx, x} with opt-level "+str(OL))

UU = transpile(R2b, basis_gates=['cx','id','rz','sx','x'], optimization_level=OL)

print("  * Number of gates:", format_val(len(UU)))
print("  * Depth:", format_val(UU.depth()))

#######################################
N = 400000
print()
print(format_head("SIMULATION","Aer.get_backend(..).run(..)"))
print("  * With aer_simulator and "+str(N)+" shots")

sim = Aer.get_backend('aer_simulator')
j = sim.run(assemble(UU,shots=N))
R = j.result().get_counts()

Y = list(itertools.product([0, 1], repeat=n))

P = np.zeros(dim)

for i,y in enumerate(Y):
	s = ''
	for b in y:
		s += str(b)
	s0 = '0'*d + '0' + s
	s1 = '0'*d + '1' + s
	P[i] = R[s0] + R[s1]
	
Z = np.sum(P)
P = P/Z

print('CIRCUIT ('+str(N)+' samples)',P)

lnZ = np.log(np.trace(R0))
Q = np.diag(R0/np.exp(lnZ))

print('GROUND TRUTH',Q)

def fidelity(P,Q):
	F = 0
	for i in range(len(P)):
		F += np.sqrt(P[i] * Q[i])
	return F**2
	
print('FIDELITY',fidelity(P,Q))
print('SUCCESS RATE',Z/N)
