# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Basic operators for the wilson hamiltonian"""


from qiskit_nature.second_q.operators import FermionicOp, SpinOp
from qiskit_nature.second_q.properties.lattices import HyperCubicLattice
import numpy as np


class FermionicSpinor:

    def __init__(self,ncomponents: int, lattice: HyperCubicLattice) -> None:
        self.ncomponents = ncomponents
        self.lattice = lattice

    def spinor_product(self,site_left:int, site_right:int, operator:np.ndarray):
        """
        Args:
            site_left: Site of the leftmost spinor.
            site_right: Site of the rightmost spinor.
            operator: Operator to to do the tensor product of both spinors.
                In this case it will be some gamma matrix.
        """
        fermionic_sum = []
        for (a,b),v in np.ndenumerate(operator):
            # print(a,b,v)
            if v != 0:
                index_A = self.ncomponents * site_left + a
                index_B = self.ncomponents * site_right + b
                fermionic_sum.append(v * (FermionicOp(f"+_{index_A}")@FermionicOp(f"-_{index_B}")))
                # print(index_A,index_B)

        # print('end')
        return sum(fermionic_sum)

class QLM:
    def __init__(self,spin:int,edges:int) -> None:

        self.spin = spin
        self.ds = 2*spin + 1
        self.edges = edges


    def operatorU(self,edge_index):
        return (self.spin*(self.spin+1))**(-0.5) *  SpinOp(f"+_{edge_index}",spin=self.spin,register_length = self.edges)

    # def operatorE()

    def field(self,position,direction):
        return 1.0






