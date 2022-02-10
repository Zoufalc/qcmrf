import os
from qiskit import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(project='ticket')  # Substitute with your provider.

provider.runtime.delete_program('qcmrf-P8LxjPxQb8')
