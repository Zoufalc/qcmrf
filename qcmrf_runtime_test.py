from qiskit import Aer
from qiskit.providers.ibmq.runtime.utils import RuntimeEncoder, RuntimeDecoder
from qiskit.providers.ibmq.runtime import UserMessenger

import json
import qcmrf_runtime as qcmrf

def test():
	backend = Aer.get_backend('qasm_simulator')
	inputs = {
		"graphs": [[[0]],[[0,1]],[[0,1],[1,2]]],
		"thetas": None,
		"gammas": None,
		"betas": None,
		"repetitions": 10,
		"shots": 32000,
		"layout": None,
		"measurement_error_mitigation": True,
		"optimization_level": 3
	}
	user_messenger = UserMessenger()
	serialized_inputs = json.dumps(inputs, cls=RuntimeEncoder)
	deserialized_inputs = json.loads(serialized_inputs, cls=RuntimeDecoder)
	qcmrf.main(backend, user_messenger, **deserialized_inputs)

test()
