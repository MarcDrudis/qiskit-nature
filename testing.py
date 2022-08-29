from fractions import Fraction
from qiskit_nature.second_q.properties import lattice_model
from qiskit_nature.second_q.properties.lattices import Lattice, HyperCubicLattice
from qiskit_nature.second_q.hamiltonians.basic_operators import FermionicSpinor
from qiskit_nature.second_q.operators import FermionicOp, SpinOp

from qiskit_nature.second_q.hamiltonians.wilson_sun_hamiltonian import WilsonModel
import matplotlib.pyplot as plt
import numpy as np



some_lattice = HyperCubicLattice((4,3,7),edge_parameter = (1.0, 2.0, 3.0),self_loops=False)



representation =[
    np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]]),
    np.array([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]]),
    np.array([[0,0,0,-1j],[0,0,1j,0],[0,1j,0,0],[-1j,0,0,0]]),
    np.array([[0,0,1,0],[0,0,0,-1],[-1,0,0,0],[0,1,0,0]])
]


w_model = WilsonModel(  lattice = some_lattice,
                        a=1,
                        r=1,
                        mass=1,
                        ncomponents=4,
                        representation=representation,
                        flavours=1,
                        spin=3)

print(w_model.hopping_term())


