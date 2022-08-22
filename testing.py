from qiskit_nature.second_q.properties import lattice_model
from qiskit_nature.second_q.properties.lattices import Lattice, HyperCubicLattice
from qiskit_nature.second_q.hamiltonians.basic_operators import FermionicSpinor
from qiskit_nature.second_q.operators import FermionicOp

from qiskit_nature.second_q.hamiltonians.wilson_sun_hamiltonian import WilsonModel
import matplotlib.pyplot as plt
import numpy as np

some_lattice = HyperCubicLattice((4,4,5))
for edges in some_lattice.weighted_edge_list:
    print(edges)


# ferm = FermionicSpinor(ncomponents=2,lattice = some_lattice)

# gamma0 = np.diag([1,-1])
# gamma1 = np.array([[0,1],[1,0]])
# gamma2 = 1.0j * gamma0 @ gamma1

