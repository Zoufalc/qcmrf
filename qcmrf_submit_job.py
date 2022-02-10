import numpy as np
from qiskit import IBMQ
import argparse

parser = argparse.ArgumentParser(description='Enqueue QCMRF experiments on qiskit runtime systems.')
parser.add_argument('backend', metavar='backend', type=str, nargs=1, help='Qiskit backend.')
args = parser.parse_args()

options = {
	'backend_name': args.backend[0]
}

runtime_inputs = {
		#"graphs": [[[0]],[[0,1]],[[0,1],[1,2]],[[0,1],[1,2],[2,3]],[[0,1],[1,2],[2,3],[0,3]],[[0,1,2,3]]],
		"graphs": [[[0]],[[0,1]],[[0,1],[1,2]],[[0,1],[1,2],[2,3]]],
		"thetas": None,
		"gammas": None,
		"betas": None,
		"repetitions": 10,
		"shots": 32000,
		"layout": None,
		"measurement_error_mitigation": True,
		"optimization_level": 3
	}

IBMQ.load_account()

provider = IBMQ.get_provider(
	hub='ibm-q-fraunhofer',
	group='fhg-all',
	project='ticket'
)

job = provider.runtime.run(
	program_id='qcmrf-P8LxjPxQb8',
	options=options,
	inputs=runtime_inputs
)

# Job id
print(job.job_id())
# See job status
print(job.status())

# Get results
result = job.result()
