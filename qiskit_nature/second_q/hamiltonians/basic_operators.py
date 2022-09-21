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

from turtle import left, right
from typing import List
from qiskit_nature.second_q.operators import FermionicOp, SpinOp
from qiskit_nature.second_q.properties.lattices import HyperCubicLattice
import numpy as np


class FermionicSpinor:
    def __init__(self, ncomponents: int, lattice: HyperCubicLattice) -> None:
        self.ncomponents = ncomponents
        self.lattice = lattice
        self.register_lenght = lattice.num_nodes * ncomponents

    def spinor_product(self, site_left: int, site_right: int, operator: np.ndarray | None):
        """
        Args:
            site_left: Site of the leftmost spinor.
            site_right: Site of the rightmost spinor.
            operator: Operator to to do the tensor product of both spinors.
                In this case it will be some gamma matrix.
        """
        if operator is None:
            operator = np.eye(N=self.ncomponents, M=self.ncomponents)
        fermionic_sum = []
        for (a, b), v in np.ndenumerate(operator):
            if v != 0:
                index_A = self.ncomponents * site_left + a
                index_B = self.ncomponents * site_right + b
                fermionic_sum.append(
                    v
                    * (
                        FermionicOp(f"+_{index_A}", register_length=self.register_lenght)
                        @ FermionicOp(f"-_{index_B}", register_length=self.register_lenght)
                    )
                )

        # Here I should get rid of the additional phase
        return sum(fermionic_sum)# * -1.0j

    def idnty(self):
        return FermionicOp("",register_length=self.register_lenght)


class QLM:
    def __init__(
        self, spin: int, lattice: HyperCubicLattice, e_value: float, electric_field: List[float]
    ) -> None:

        self.spin = spin
        self.ds = 2 * spin + 1
        self.lattice = lattice
        self.edges = len(lattice.weighted_edge_list)
        self.e = e_value
        self.electric_field = electric_field

    def idnty(self):
        return SpinOp("I_0", spin=self.spin, register_length=self.edges)

    def operatorU(self, edge_index):
        return (self.spin * (self.spin + 1)) ** (-0.5) * SpinOp(
            f"+_{edge_index}", spin=self.spin, register_length=self.edges
        )

    def operatorE(self, edge_index):
        return self.e * SpinOp(f"Z_{edge_index}", spin=self.spin, register_length=self.edges)

    def operatorE_2(self, edge_index):
        return (self.e**2) * SpinOp(f"Z_{edge_index}^2", spin=self.spin, register_length=self.edges)

    def operator_plaquette(self, node_a: int, node_b: int, node_c: int, node_d: int):
        return (self.spin * (self.spin + 1)) ** (-2) * SpinOp(
            f"+_{node_a} +_{node_b} -_{node_c} -_{node_d}",
            spin=self.spin,
            register_length=self.edges,
        )

    def operator_divergence(self, node):
        indexed_graph = self.lattice.indexed_graph()
        divergence = []
        for neighbor in self.lattice.graph.neighbors(node):
            directn = self.lattice.direction((node,neighbor))
            edge_index = indexed_graph.get_edge_data(node,neighbor)
            divergence.append(np.sign(-directn)*self.operatorE(edge_index))
        return sum(divergence)


        # #Still not good enough for boundary conditions
        # divergence = []


        # #This dictionary indexes the coefficient that we need to add to a given operator depending on
        # #whether a given edge it's at its right(-) or left(+).
        # coeff_sign = {(-1,0):+1,(1,0):-1,(-1,1):-1,(1,1):+1}

        # for bound_direction in self.lattice.base_connections():
        #     for periodic,direction in enumerate(bound_direction):
        #         for sign in [+1,-1]:
        #             if 0<=node +sign*direction<=self.lattice.num_nodes and indexed_graph.has_edge(node,node+ sign * direction):
        #                 edge_index = indexed_graph.get_edge_data(node, node +sign*direction)
        #                 divergence.append(coeff_sign[(sign,periodic)]*self.operatorE(edge_index))

        # return sum(divergence)
