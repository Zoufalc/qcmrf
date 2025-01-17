import sys
import numpy as np
from scipy.linalg import expm

np.set_printoptions(threshold=sys.maxsize, linewidth=1024)

import itertools
from colored import fg, bg, attr

from qiskit.opflow import I, X, Z, Plus, Minus, H, Zero, One, MatrixOp, OperatorBase
from qiskit.compiler import transpile
from qiskit import QuantumRegister, QuantumCircuit
from qiskit import Aer, assemble


#######################################

# Compute fidelity between probability mass functions, given by P and Q
# P[i] and Q[i] contains probabilities for the i-th event
def fidelity(P, Q):
    F = 0
    for i in range(len(P)):
        F += np.sqrt(P[i] * Q[i])
    return F ** 2


# Compute Kullback-Leibler between probability mass functions, given by P and Q
# P[i] and Q[i] contains probabilities for the i-th event
def KL(P, Q):
    kl = 0
    for i in range(len(P)):
        if Q[i] > 0 and P[i] > 0:
            kl += P[i] * np.log(P[i] / Q[i])
    return kl


# Python helper for iterating over tuples of length n
def grouped(iterable, n):
    return zip(*[iter(iterable)] * n)


def merge(L):
    R = []
    for (M0, M1) in grouped(L, 2):
        R.append((((I + Z) / 2) ^ M0) + (((I - Z) / 2) ^ M1))
    return R


def merge_all(L):
    R = L
    while len(R) > 1:
        R = merge(R)
    return R[0]


turquoise = '#179c7d'
orange = '#eb6a0a'
blue = '#1f82c0'


# string formatting
def format_head(h, t):
    return fg(turquoise) + attr('bold') + h + attr('reset') + ': ' + fg('white') + attr(
        'bold') + t + attr('reset')


# string formatting
def format_val(val):
    if type(val) is int:
        return fg('white') + attr('bold') + str(val) + attr('reset')
    else:
        return fg('white') + attr('bold') + np.format_float_scientific(val, precision=4) + attr(
            'reset')


# string formatting
def format_math(math):
    brackets = ['(', ')']
    operators = ['+', '-', '*', '/', '_', '^']
    mcol = turquoise
    bcol = 'white'
    ocol = orange
    for b in brackets:
        math = math.replace(b, fg(bcol) + b + fg(mcol))
    for o in operators:
        math = math.replace(o, fg(ocol) + o + fg(mcol))
    return fg(mcol) + math + attr('reset')


####################################### model funcs

# compute 2**n X 2**n binary diaginal matrix D
# D_ii is 1 if, for the i-th basis state x**(i), we have x**(i)_C = y
def genPhi(c, y):
    result = 1

    plus = [v for i, v in enumerate(c) if not y[i]]  # 0
    minus = [v for i, v in enumerate(c) if y[i]]  # 1

    for i in range(n):
        f = I
        if i in minus:
            f = (I - Z) / 2
        elif i in plus:
            f = (I + Z) / 2

        result = result ^ f

    return result


# return pair R,L
# R is the Hamiltoian sum_C sum_y theta_C,y * Phi_C,y
# L list of lists, one list per clique C
# List for clique C contains pairs (theta,Phi) of weight theta_C,y and Phi_C,y
def genHamiltonian():
    L = []
    R = 0
    i = 0
    for l, c in enumerate(C):
        Ly = []
        for r, y in enumerate(list(itertools.product([0, 1], repeat=len(c)))):
            Phi = genPhi(c, y)
            theta = np.random.uniform(low=-5.0, high=-0.001)  # we need a negative MRF
            R += Phi * -theta
            Ly.append((theta, Phi))  # list of all factors that belong to same clique
            i = i + 1
        L.append(Ly)  # list of factor-lists, one factor-list per clique

    return R, L


# compute parameters of RZ-gates
def genPhaseFactors(ey):
    t = np.sqrt((-1 - ey) / (-1 + ey))

    phi1 = np.angle(((1 + t * 1j) + ey * (1 - t * 1j)) / 2)
    phi2 = np.real(np.angle(np.sqrt(0.5) * np.sqrt(1 - ey) * (t + 1j)))

    return np.array([phi1, phi2])


# compute unitary U**gamma = exp(U)
def genUphi(U, phi):
    RZ1 = ((-1) * phi[0] * Z).exp_i() ^ (I ^ n)  # ignored sign of phi
    RZ2 = ((-1) * phi[1] * Z).exp_i() ^ (I ^ n)  # ignored sign of phi
    assert np.isclose(phi[0], phi[1])
    # assert U == ~U (always true)
    # assert RZ1 == RZ2 (always true)
    return (RZ1 @ U) ** 2  # ignored conjugate transpose of first U bc symmetric and real


# returns unitary if Eigenvalues of A are bounded by 1
def uniEmbedding(A):
    return (X ^ ((I ^ n) - A)) + (Z ^ A)


def uniEmbeddingN(A):
    M = A.to_matrix()
    U = np.matrix(
        np.block([[M, np.sqrt(np.eye(2 ** n) - (M @ M))], [np.sqrt(np.eye(2 ** n) - (M @ M)), -M]]))
    return MatrixOp(U)


# returns unitary if A is unitary
def conjugateBlocks(A):
    return (((I + Z) / 2) ^ A) + (((I - Z) / 2) ^ (~A))


# works only if number of cliques is a power of two
def expH_from_list_blocked(beta, L0, lnZ=0):
    R = []
    for Ly in L0:
        L = [genUphi(uniEmbedding(Phi0), genPhaseFactors(np.exp(beta * (w0 - (lnZ / len(C)))))) for
             w0, Phi0 in Ly]
        R.append(merge_all(L))
    M = merge_all(R)
    O = H ^ (I ^ int(n + 1 + np.log2(d)))
    return O @ conjugateBlocks(M) @ (~O)


def expH_from_list_real_algebraic(beta, L0, lnZ=0):
    RESULT = I ^ (n + 1)
    for L in L0:
        for (w0, Phi0) in L:
            U = genUphi(uniEmbedding(Phi0), genPhaseFactors(np.exp(beta * (w0 - (lnZ / len(C))))))
            RESULT = (U + (~U)) / 2 @ RESULT  # non-unitary extraction of real part
    return RESULT


def expH_from_list_unreal(beta, L0, lnZ=0):
    qr = QuantumRegister(n + 2, 'q')  # one aux for unitary embedding plus one aux per factor

    # create empty main circuit with d+1 aux qubits
    circ = QuantumCircuit(n + 2, n + 2)
    for i in range(n + 1):
        circ.h(qr[i])

    RESULT = I ^ (n + 1)
    for L in L0:
        for (w0, Phi0) in L:
            w = 0.5 * w0 - (lnZ / len(C))  # 0.5 => sqrt(exp(..))
            assert w < 0

            U = genUphi(uniEmbedding(Phi0), genPhaseFactors(np.exp(beta * w)))
            RESULT = U @ RESULT  # complex result

    u = conjugateBlocks(RESULT).to_circuit().to_instruction(label='MRF')

    circ.h(qr[n + 1])
    circ.append(u, range(n + 2))
    circ.h(qr[n + 1])

    circ.measure(range(n + 2), range(n + 2))

    return circ


# core method
# computes exp(-beta H) = PROD_j exp(-beta w_j Phi_j) = PROD_j REAL( P**(beta w_j)(U_j) )
def expH_from_list_real_RUS(beta, L0, lnZ=0):
    CL = len(L0)
    qr = QuantumRegister(n + CL, 'q')  # one aux for unitary embedding plus one aux per factor

    # create empty main circuit with d+1 aux qubits
    circ = QuantumCircuit(n + CL, n + CL)
    for i in range(n):
        circ.h(qr[i])

    circ.barrier(qr)

    i = 0  # enumerate 0..CL-1
    for ii, L in enumerate(L0):
        RESULT = I ^ (n + 1)
        for jj, (w0, Phi0) in enumerate(L):
            w = 0.5 * w0 - (lnZ / len(C))  # 0.5 => sqrt(exp(..))
            assert w < 0

            # compute U**gamma = P**(beta w_j)(U_j)
            U = genUphi(uniEmbedding(Phi0), genPhaseFactors(np.exp(beta * w)))
            RESULT = U @ RESULT

        M = RESULT.to_matrix()[:2 ** (n), :2 ** (n)]  # hack
        RESULT = MatrixOp(M)
        # print(M @ np.conjugate(np.transpose(M)))
        #		print(RESULT.to_matrix().astype(float))

        # Write U**gamma and ~U**gamma on diagonal of matrix, creates j-th aux qubit
        # Create "instruction" which can be used in another circuit
        u = conjugateBlocks(RESULT)
        u = aqc_compil(u)
        # u = u.to_circuit().to_instruction(label='U_C' + str(ii))

        # add Hadamard to j-th aux qubit
        circ.h(qr[n + i])
        # add U**gamma circuit to main circuit
        circ.append(u, [qr[j] for j in range(n)] + [qr[n + i]])
        # add another Hadamard to aux qubit
        circ.h(qr[n + i])
        circ.measure([n + i], [n + i])

        circ.barrier(qr)

        i = i + 1

    # measure all qubits
    circ.measure(range(n), range(n))
    return circ


#######################################


def aqc_compil(circuit: OperatorBase):
    from qiskit.algorithms.optimizers import L_BFGS_B
    from qiskit.transpiler.synthesis.aqc.aqc import AQC
    from qiskit.transpiler.synthesis.aqc.cnot_structures import make_cnot_network
    from qiskit.transpiler.synthesis.aqc.cnot_unit_circuit import CNOTUnitCircuit
    from qiskit.transpiler.synthesis.aqc.cnot_unit_objective import DefaultCNOTUnitObjective
    seed = 134

    num_qubits = circuit.num_qubits

    cnots = make_cnot_network(
        num_qubits=num_qubits, network_layout="spin", connectivity_type="full", depth=0
    )

    optimizer = L_BFGS_B(maxiter=200)

    aqc = AQC(optimizer=optimizer, seed=seed)
    target_matrix = np.array(circuit.to_matrix())
    approximate_circuit = CNOTUnitCircuit(num_qubits, cnots)
    approximating_objective = DefaultCNOTUnitObjective(num_qubits, cnots)
    # todo: approximating_objective = DefaultCNOTUnitObjective(approximate_circuit)

    aqc.compile_unitary(
        target_matrix=target_matrix,
        approximate_circuit=approximate_circuit,
        approximating_objective=approximating_objective,
    )
    # print(approximate_circuit)
    # print('Depth ', approximate_circuit.depth())
    return approximate_circuit.to_instruction()


RUNS = [[[0]], [[0, 1]], [[0, 1], [1, 2]], [[0, 1], [1, 2], [2, 3]]]
# ,[[0,1],[1,2],[2,3],[0,3]],[[0,1,2],[0,2,3]],[[0,1,2,3]]]
# RUNS = [[[0,1],[1,2],[2,3],[0,3]],[[0,1],[1,2],[2,3],[0,3],[3,4,5]]]
# RUNS = [[[0,1,2,3],[3,4,5,6]]]

logfile = open("results_experiment_aqc.csv", "w")
logfile.write('n,d,num_cliques,C_max,fidelity,KL,success_rate,num_gates,depth,shots,w_min,w_max\n')

for C in RUNS:
    for II in range(10):
        n = len(np.unique(np.array(C).flatten()))  # number of (qu)bits
        d = 0
        cmax = 0
        for c in C:
            m = len(c)
            if m > cmax:
                cmax = m
            d = d + (2 ** m)

        dim = 2 ** n

        HAM, LL = genHamiltonian()  # L is list of factors
        beta = 1
        R0 = expm(-beta * HAM.to_matrix())  # exp(-βH) via numpy for debugging
        R2b = expH_from_list_real_RUS(beta, LL)
        OL = 3
        UU = transpile(R2b, basis_gates=['cx', 'id', 'rz', 'sx', 'x'], optimization_level=OL)
        # print(UU)
        N = 1000000
        sim = Aer.get_backend('aer_simulator')
        j = sim.run(assemble(UU, shots=N))
        R = j.result().get_counts()
        print(R)
        Y = list(itertools.product([0, 1], repeat=n))
        P = np.zeros(dim)

        wmin = None
        wmax = None
        for l1 in LL:
            for ww, ll in l1:
                if wmin is None:
                    wmin = ww
                elif ww < wmin:
                    wmin = ww

                if wmax is None:
                    wmax = ww
                elif ww > wmax:
                    wmax = ww

        for i, y in enumerate(Y):
            s = ''
            for b in y:
                s += str(b)
            s0 = '0' * len(C) + s

            if s0 in R:
                P[i] += R[s0]

        ZZ = np.sum(P)
        P = P / ZZ

        lnZ = np.log(np.trace(R0))
        Q = np.diag(R0 / np.exp(lnZ))

        logs = str(n) + ',' + str(d) + ',' + str(len(C)) + ',' + str(cmax) + ',' + str(
            np.real(fidelity(P, Q))) + ',' + str(np.real(KL(Q, P))) + ',' + str(ZZ / N) + ',' + str(
            len(UU)) + ',' + str(UU.depth()) + ',' + str(N) + ',' + str(wmin) + ',' + str(wmax)
        print(logs)
        logfile.write(logs + '\n')

logfile.close()
