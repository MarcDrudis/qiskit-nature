from fractions import Fraction
from itertools import product
from qiskit_nature.second_q.properties import LatticeDrawStyle
from qiskit_nature.second_q.properties.lattices import Lattice, HyperCubicLattice
from qiskit_nature.second_q.hamiltonians.basic_operators import FermionicSpinor
from qiskit_nature.second_q.operators import FermionicOp, SpinOp, MixedOp
from qiskit_nature.second_q.mappers import QubitConverter, JordanWignerMapper, LogarithmicMapper, FermionicMapper

from qiskit_nature.second_q.hamiltonians.wilson_sun_hamiltonian import WilsonModel
import matplotlib.pyplot as plt
import numpy as np
import time

from qiskit_nature.second_q.properties.lattices import BoundaryCondition






some_lattice = HyperCubicLattice((5,),self_loops=False,boundary_condition=BoundaryCondition.OPEN)





# print(len(some_lattice.get_plaquettes()))
# print(some_lattice.graph.has_edge(7,9))
# print(list(some_lattice.directions.keys()))

# for index in some_lattice.graph.edge_indices():
#     some_lattice._graph.update_edge_by_index(index,index)

# some_lattice.draw(style = LatticeDrawStyle(with_labels=True,edge_labels=str))
# plt.show()



representation3 = [
    np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]]),
    np.array([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]]),
    np.array([[0,0,0,-1j],[0,0,1j,0],[0,1j,0,0],[-1j,0,0,0]]),
    np.array([[0,0,1,0],[0,0,0,-1],[-1,0,0,0],[0,1,0,0]])
]

representation2 = [
    np.array([[0,1],[1,0]]),
    np.array([[0,1j],[-1j,0]]),
    np.array([[1,0],[0,-1]]),
]




w_model = WilsonModel(  lattice = some_lattice,
                        a=1,
                        r=1,
                        mass=1,
                        representation=representation2,
                        flavours=1,
                        spin=3,
                        electric_field=(1,2,3),
                        e_value = 0.025,
                        q=1,
                        lmbda=40
)






qubit_converter = QubitConverter(mappers = [JordanWignerMapper(),LogarithmicMapper()])
# print(qubit_converter.convert(w_model.mass_term()))
idntyspin =w_model._QLM_spin.idnty(2)
idntyferm = w_model._fermionic_spinor.idnty()
print(idntyspin)
print(qubit_converter.convert(MixedOp(([idntyferm,idntyspin],1.0))))











