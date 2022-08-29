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

"""The Fermi-Hubbard model"""
import numpy as np
from qiskit_nature.second_q.operators.spin_op import SpinOp

# from qiskit_nature.second_q.operators import MixedOp
from qiskit_nature.second_q.properties.lattices import Lattice

from qiskit_nature.second_q.properties import LatticeModel
from qiskit_nature.second_q.operators import FermionicOp

from ..properties.lattices.hyper_cubic_lattice import HyperCubicLattice
from .basic_operators import FermionicSpinor, QLM

from typing import List


class WilsonModel(LatticeModel):
    """The Wilson Model model."""

    def __init__(
        self,
        lattice: HyperCubicLattice,
        a: complex,
        r: complex,
        mass: float,
        ncomponents: int,
        representation: List[np.ndarray],
        flavours: int,
        spin: int,
        symmetry: str = "???",
        encoding="quantum_link",
    ):
        """
        Args:
            lattice: Lattice on which the model is defined.
            a: Lattice Constant
            r: Wilson Parameter
            mass: Mass
            flavours: Number of flavours
            spin: Twice the value of the spin for the field operator in the Quantum Link Model.
            symmetry: Symmetry of the bosonic model. Can be U(1) or SU(n).
            encoding: The encoding of the model. Can only be "quantum_link".
        """
        super().__init__(lattice)
        self._a = a
        self._r = r
        self._mass = mass
        self._lattice = lattice
        self._d = lattice.dim
        self.representation = representation
        self.ncomponents = ncomponents
        self._fermionic_spinor = FermionicSpinor(ncomponents=ncomponents,lattice=lattice)
        self._QLM_spin = QLM(spin=spin,edges = len(self.lattice.weighted_edge_list))



    def mass_term(self):
        """Creates the mass term for the wilson hamiltonian."""
        mass_terms = []
        for site in self.lattice.node_indexes:
            mass_terms.append(self._fermionic_spinor.spinor_product(site,site,self.representation[0]))
        return (self._mass+self._r*self._d/self._a) * sum(mass_terms)

    def hopping_term(self):
        """Creates the hopping term in the hamiltonian."""
        hopping_terms = []

        x_dim,y_dim,_ = self.lattice.size
        directions = {  1:1,
                        x_dim:2,
                        x_dim*y_dim:3
                        }
        tensor_size = self.representation[0].shape[0]
        for edge_index, edge in enumerate(self.lattice.weighted_edge_list):
            k = directions[np.abs(edge[1]-edge[0])]
            fermionic_tensor = self.representation[0] @ (self._r * np.eye(tensor_size) + 1.0j*self.representation[k])
            fermionic_part = self._fermionic_spinor.spinor_product(edge[0],edge[1],fermionic_tensor)

            bosonic_part = self._QLM_spin.operatorU(edge_index=edge_index)
            bosonic_part = self._QLM_spin.field(None,None) * bosonic_part #This is just conceptual

            # mixed_op = MixedOP(coeff = 1.0, fermionic = fermionic_part, bosonic = bosonic_part)
            hopping_terms.append((fermionic_part,bosonic_part))

        return hopping_terms



    def second_q_ops(self, display_format: str = "sparse") :
        """Return the Hamiltonian of the Wilson Model in terms of `MixedOp.

        Args:
            display_format: If sparse, the label is represented sparsely during output.
                If dense, the label is represented densely during output. Defaults to "dense".

        Returns:
            MixedOp: The Hamiltonian of the Wilson Model.
        """


        pass
