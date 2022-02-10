#
# (C) Copyright Nico Piatkowski and Christa Zoufal 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A self-contained QCMRF."""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.linalg import expm
import itertools

import time

from qiskit.opflow import I, X, Z, MatrixOp
from qiskit.compiler import transpile
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit.utils import QuantumInstance

class Publisher:
	"""Class used to publish interim results."""

	def __init__(self, messenger):
		self._messenger = messenger

	def callback(self, *args, **kwargs):
		text = list(args)
		for k, v in kwargs.items():
			text.append({k: v})
		self._messenger.publish(text)


# begin QCMRF

class QCMRF(QuantumCircuit):
	"""Quantum circuit Markov random field."""

	def __init__(
		self,
		cliques=None,
		theta=None,
		gamma=None,
		beta : float = 1,
		name: str = "QCMRF"
	):
		r"""
		Args:
			cliques (List[List[int]]): List of integer lists, representing the clique structure of the
				Markov random field. For a n-dimensional random field, variable indices 0..n-1 are required.
			theta (List[float], optional): The native parameter of the Markov random field.
			gamma (List[float], optional): The alternative circuit parameters of the Markov random field.
			beta (float, optional): inverse temperature, default 1
			name (str): A name of the circuit, default 'QCMRF'
		"""
		self._cliques = cliques
		self._theta = theta
		self._gamma = gamma
		self._beta = beta
		self._name = name
		
		if type(self._cliques) != list or type(self._cliques[0]) != list or type(self._cliques[0][0]) != int:
			raise ValueError(
				"The set of clique is not set properly. Type must be list of list of int."
			)
		
		self._num_cliques = len(self._cliques)
		self._n = self._cliques[0][0] # first vertex of first clique
		
		for C in cliques:
			for v in C:
				if v > self._n:
					self._n = v
		self._n += 1
		
		self._dim = 0
		self._c_max = 0
		for C in self._cliques:
			m = len(C)
			if m > self._c_max:
				self._c_max = m
			self._dim += (2**m)
			
			
		if self._theta is not None and len(self._theta) != self._dim:
			raise ValueError(
				"The MRF parameter vector has an incorrect dimension. Expected: " + str(self._dim)
			)

		if self._gamma is not None and len(self._gamma) != self._dim:
			raise ValueError(
				"The QCMRF parameter vector has an incorrect dimension. Expected: " + str(self._dim)
			)

		super().__init__(self._n + self._num_cliques, self._n + self._num_cliques, name=name)
		
		self._build()

	@property
	def dimension(self):
		"""The parameter dimension of the Markov random field.

		Returns:
			 int: number of parameters.
		"""
		return self._dim

	@property
	def cliques(self):
		"""Returns the clique set of the Markov random field.

		Returns:
			List[List[int]]: cliques.
		"""
		return self._cliques
		
	@property
	def num_vertices(self):
		"""Returns the number of variables of the Markov random field.

		Returns:
			int: number of vertices.
		"""
		return self._n
		
	@property
	def num_cliques(self):
		"""Returns the number of cliques of the Markov random field.

		Returns:
			int: number of cliques.
		"""
		return len(self._cliques)

	@property
	def max_clique(self):
		"""Returns the size of the larges clique of the Markov random field.

		Returns:
			int: size of largest clique.
		"""
		return self._c_max

	def sufficient_statistic(self,C,y):
		"""Returns the Pauli-Markov sufficient statistic for clique-state pair (C,y)

		Returns:
			PauliSumOp: opflow representation of Pauli-Markov sufficient statistic
		"""
		result = 1

		plus  = [v for i,v in enumerate(C) if not y[i]] # 0
		minus = [v for i,v in enumerate(C) if     y[i]] # 1)

		for i in range(self._n):
			f = I
			if i in minus:
				f = (I-Z)/2
			elif i in plus:
				f = (I+Z)/2

			result = result^f

		return result
		
	def groundtruthHamiltonian(self):
		H = 0
		i = 0
		for C in self._cliques:
			for y in list(itertools.product([0, 1], repeat=len(C))):
				Phi = self.sufficient_statistic(C,y)
				
				if self._theta is not None:
					w = self._beta * self._theta[i]
				elif self._gamma is not None:
					w = 2*np.log(np.cos(self._gamma[i]))
				
				H += Phi * -w
				i += 1
		return H

	def _conjugateBlocks(self,A):
		"""Returns a block unitary with A and ~A on its diagonal."""
		return (((I+Z)/2)^A) + (((I-Z)/2)^(~A))

	def _build(self):
		"""Return the actual QCMRF."""

		for i in range(self._n):
			self.h(i)
		self.barrier()
		
		if self._theta is None and self._gamma is None:
			self._theta = []
			for i in range(self._dim):
				self._theta.append(np.random.uniform(low=-5.0,high=-0.001))

		i = 0
		for ii,C in enumerate(self._cliques):
			factor = I^(self._n+1)
			for y in list(itertools.product([0, 1], repeat=len(C))):
				Phi = self.sufficient_statistic(C,y)
				U = (X^((I^(self._n))-Phi)) + (Z^Phi)

				if self._theta is not None:
					w = 0.5 * np.arccos(np.exp(self._beta*0.5*self._theta[i]))
				elif self._gamma is not None:
					w = self._gamma[i]
				i += 1

				RZ = (-w * Z).exp_i() ^ (I^self._n)
				Ugam = (RZ @ U)**2
				factor = Ugam @ factor

			M = factor.to_matrix()[:2**(self._n),:2**(self._n)] # hack to reduce aux qubits by 1, works only for small n
			factor = MatrixOp(M)
			
			# Create "instruction" which can be used in another circuit
			u = self._conjugateBlocks(factor).to_circuit().to_instruction(label='U_C('+str(ii)+')')

			# RUS for real part extraction
			self.h(self._n + ii)
			self.append(u, [j for j in range(self._n)]+[self._n + ii])
			self.h(self._n + ii)
			self.measure(self._n + ii, self._n + ii) # real part extraction successful when measure 0

			self.barrier()
		self.measure(range(self._n),range(self._n))
			
# end QCMRF

def fidelity(P,Q):
	"""Returns fidelity between probability mass functions, given by P and Q."""
	F = 0
	for i in range(len(P)):
		F += np.sqrt(P[i] * Q[i])
	return F**2

def KL(P,Q):
	"""Returns Kullback-Leibler divergence between probability mass functions, given by P and Q."""
	kl = 0
	for i in range(len(P)):
		if Q[i] > 0 and P[i] > 0:
			kl += P[i] * np.log(P[i] / Q[i])
	return kl

def extract_probs(R,n,a):
	Y = list(itertools.product([0, 1], repeat=n))
	P = np.zeros(2**n)
	for i,y in enumerate(Y):
		s = ''
		for b in y:
			s += str(b)
		s0 = '0'*a + s

		if s0 in R:
			P[i] += R[s0]
			
	z = np.sum(P)
	
	return P/z, z

def run(backend,graphs,thetas,gammas,betas,repetitions,shots,layout=None,callback=None,measurement_error_mitigation=True,optimization_level=3):
	for i in range(len(graphs)):
		for j in range(repetitions):
			if betas is not None:
				b = betas[i]
			else:
				b = 1

			if thetas is not None:
				C = QCMRF(graphs[i],theta=thetas[i],beta=b)
			elif gammas is not None:
				C = QCMRF(graphs[i],gamma=gammas[i],beta=b)
			else:
				C = QCMRF(graphs[i],beta=b)

		
			if not measurement_error_mitigation:
				s1 = time.time()
				T = transpile(C, backend, optimization_level=optimization_level, seed_transpiler=42) # initial_layout=layout
				s1 = time.time() - s1
				s2 = time.time()
				job = backend.run(T, shots=shots)
				result = job.result()
				
			else:
				s1 = time.time()
				qi = QuantumInstance(backend=backend, shots=shots, optimization_level=optimization_level, seed_transpiler=42, skip_qobj_validation=False, measurement_error_mitigation_cls=CompleteMeasFitter, measurement_error_mitigation_shots=shots/2)
				T = qi.transpile([C])[0]
				s1 = time.time() - s1
				s2 = time.time()
				result = qi.execute([T])
			s2 = time.time() - s2
			
			#rjob = backend.retrieve_job(job.job_id())
			#rjob.wait_for_final_state()
			s3 = time.time()
			P, c = extract_probs(result.get_counts(), C.num_vertices, C.num_cliques)
			s3 = time.time() - s3
			
			H = C.groundtruthHamiltonian()
			RHO = expm(-b*H.to_matrix())
			z = np.trace(RHO)
			Q = np.diag(RHO)/z
			
			callback(C.num_vertices, C.dimension, C.num_cliques, C.max_clique, np.real(fidelity(P,Q)), np.real(KL(Q,P)), c/shots, len(T), T.depth(), shots, s1, s2, s3)

def main(backend, user_messenger, **kwargs):
	"""Entry function."""
	# parse inputs
	mandatory = {"graphs"}
	missing = mandatory - set(kwargs.keys())
	if len(missing) > 0:
		raise ValueError(f"The following mandatory arguments are missing: {missing}.")

	# Extract the input form the kwargs and build serializable kwargs for book keeping.
	serialized_inputs = {}
	serialized_inputs["graphs"] = kwargs["graphs"]
	if kwargs["thetas"] is not None:
		serialized_inputs["thetas"] = kwargs["thetas"]
	if kwargs["gammas"] is not None:
		serialized_inputs["gammas"] = kwargs["gammas"]
	if kwargs["betas"] is not None:
		serialized_inputs["betas"] = kwargs["betas"]

	repetitions = kwargs.get("repetitions", 1)
	serialized_inputs["repetitions"] = repetitions

	shots = kwargs.get("shots", 8192)
	serialized_inputs["shots"] = shots

	measurement_error_mitigation = kwargs.get("measurement_error_mitigation", False)
	serialized_inputs["measurement_error_mitigation"] = measurement_error_mitigation

	optimization_level = kwargs.get("optimization_level", 3)
	serialized_inputs["optimization_level"] = optimization_level
	
	layout = kwargs.get("layout", None)
	serialized_inputs["layout"] = layout

	# publisher for user-server communication
	publisher = Publisher(user_messenger)
	
	# dictionary to store the history of the runs
	history = {"n": [], "d": [], "num_cliques": [], "max_clique": [], "fidelity": [], "KL": [], "success_rate": [], "gates": [], "depth": [], "shots": [], "transpile_time": [], "prepare_time": [], "exec_time": []}

	def store_history_and_forward(n, d, num_cliques, max_clique, fidelity, KL, success_rate, gates, depth, shots, transpile_time, prepare_time, exec_time):
		# store information
		history["n"].append(n)
		history["d"].append(d)
		history["num_cliques"].append(num_cliques)
		history["max_clique"].append(max_clique)
		history["fidelity"].append(fidelity)
		history["KL"].append(KL)
		history["success_rate"].append(success_rate)
		history["gates"].append(gates)
		history["depth"].append(depth)
		history["shots"].append(shots)
		history["transpile_time"].append(transpile_time)
		history["prepare_time"].append(prepare_time)
		history["exec_time"].append(exec_time)
		# and forward information to users callback
		publisher.callback(n, d, num_cliques, max_clique, fidelity, KL, success_rate, gates, depth, shots, transpile_time, prepare_time, exec_time)

	result = run(
		backend,
		kwargs["graphs"],
		kwargs["thetas"],
		kwargs["gammas"],
		kwargs["betas"],
		kwargs["repetitions"],
		kwargs["shots"],
		kwargs["layout"],
		store_history_and_forward,
		kwargs["measurement_error_mitigation"],
		kwargs["optimization_level"]
	)

	serialized_result = {
		"Fidelity_μ": np.mean(history['fidelity']),
		"Fidelity_σ": np.std(history['fidelity']),
		"KL_μ": np.mean(history['KL']),
		"KL_σ": np.std(history['KL']),
		"SR_μ": np.mean(history['success_rate']),
		"SR_σ": np.std(history['success_rate']),
		"Depth_μ": np.mean(history['depth']),
		"Depth_σ": np.std(history['depth']),
		"all_results": history,
		"inputs": serialized_inputs,
	}

	return serialized_result
