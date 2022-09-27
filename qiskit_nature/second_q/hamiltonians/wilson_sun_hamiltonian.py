# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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
from qiskit_nature.second_q.mappers.jordan_wigner_mapper import JordanWignerMapper
from qiskit_nature.second_q.mappers.logarithmic_mapper import LogarithmicMapper
from qiskit_nature.second_q.mappers.qubit_converter import QubitConverter
from qiskit_nature.second_q.operators.spin_op import SpinOp

from qiskit_nature.second_q.operators import MixedOp
from qiskit_nature.second_q.properties import LatticeModel
from qiskit_nature.second_q.operators import FermionicOp, mixed_op

from qiskit_nature.second_q.properties.lattices import HyperCubicLattice
from qiskit_nature.second_q.operators import MixedOp

from qiskit_nature.second_q.hamiltonians.basic_operators import FermionicSpinor, QLM

from typing import List


class WilsonModel(LatticeModel):
    """The Wilson Model."""

    def __init__(
        self,
        lattice: HyperCubicLattice,
        lattice_constant: complex,
        wilson_parameter: complex,
        charge: float,
        mass: float,
        constraint_coefficient: float,
        representation: List[np.ndarray],
        electric_field: List[float],
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
        self.lattice_constant = lattice_constant
        self.charge = charge
        self.wilson_parameter = wilson_parameter
        self.constraint_coefficient = constraint_coefficient
        self.mass = mass
        self._dimension = lattice.dim
        self.representation = representation
        self._ncomponents = int(2 ** ((self._d + 1) // 2))
        self.fermionic_spinor = FermionicSpinor(ncomponents=self._ncomponents, lattice=lattice)
        self.QLM_spin = QLM(
            spin=spin,
            lattice=lattice,
            charge=charge,
            electric_field=electric_field,
        )

    @property
    def lattice(self) -> HyperCubicLattice:
        """Return a copy of the input lattice."""
        return self._lattice.copy()

    def mass_term(self) -> FermionicOp:
        """Creates the mass term for the wilson hamiltonian."""
        mass_terms = []
        for site in self.lattice.node_indexes:
            mass_terms.append(
                self.fermionic_spinor.spinor_product(site, site, self.representation[0])
            )
        return (self.mass + self.wilson_parameter * self._d / self.lattice_constant) * sum(
            mass_terms
        )

    def hopping_term(self) -> MixedOp:
        """Creates the hopping term in the hamiltonian.
        Note that this only creates the first term and we still need to add the conjugate transpose.
        """
        hopping_terms = []

        tensor_size = self.representation[0].shape[0]
        for edge_index, (node_a, node_b) in enumerate(self.lattice.graph.edge_list()):
            k = self.lattice.direction((node_a, node_b))
            fermionic_tensor = self.representation[0] @ (
                self.wilson_parameter * np.eye(tensor_size) + 1.0j * self.representation[np.abs(k)]
            )
            # print(f"Tensor for nodes {node_a}, {node_b} is:")
            # print(fermionic_tensor)
            fermionic_part = self.fermionic_spinor.spinor_product(node_a, node_b, fermionic_tensor)
            # fermionic_part = self.fermionic_spinor.spinor_product(node_b, node_a, fermionic_tensor)

            bosonic_part = self.QLM_spin.operatorU(edge_index=edge_index)

            mix_op = MixedOp(([fermionic_part, bosonic_part], 1 / (2 * self.lattice_constant)))
            mix_op_dag = MixedOp(
                (
                    [fermionic_part.adjoint(), bosonic_part.adjoint()],
                    1 / (2 * self.lattice_constant),
                )
            )
            hopping_terms.append(mix_op)
            hopping_terms.append(mix_op_dag)
        return sum(hopping_terms)

    def plaquette_term(self) -> SpinOp:
        """Creates the plaquette terms for the hamiltonian."""

        if self._d == 1:
            return None

        plaquette_terms = []
        for plaquette in self.lattice.get_plaquettes():
            plaquette_terms.append(self.QLM_spin.operator_plaquette(*plaquette))
            plaquette_terms.append(self.QLM_spin.operator_plaquette(*plaquette).adjoint())

        return (-1) / (4 * self.e) * sum(plaquette_terms)

    def link_term(self) -> SpinOp:
        link_terms = []
        for edge_index, (node_a, node_b) in enumerate(self.lattice.graph.edge_list()):
            op_E = self.QLM_spin.operatorE(edge_index=edge_index)
            op_E_2 = self.QLM_spin.operatorE_2(edge_index=edge_index)
            idnty = self.QLM_spin.idnty()
            k = self.lattice.direction((node_a, node_b))
            field = self.QLM_spin.electric_field[k - 1]
            link_terms.append(op_E_2 + 2 * field * op_E + field**2 * idnty)

        return self.e**2 / 2 * sum(link_terms)

    def gauss_operators(self) -> List[MixedOp]:
        """Returns a list of the gauss operators imposing constraints over each node of the graph."""
        charge_offset = -self.e
        gauss_terms = []
        for site in self.lattice.node_indexes:
            gauss_term = MixedOp(
                ([self.fermionic_spinor.idnty(), self.QLM_spin.operator_divergence(site)], 1)
            )
            gauss_term += -self.charge * self.fermionic_spinor.spinor_product(site, site, None)
            gauss_term += MixedOp(
                ([self.fermionic_spinor.idnty(), self.QLM_spin.idnty()], charge_offset)
            )
            gauss_terms.append(gauss_term)
        return gauss_terms

    def second_q_ops(self):
        """Returns the Hamiltonian of the Wilson Model in terms of `MixedOp`."""

        mass_term = self.mass_term()
        link_term = self.link_term()
        hopping_term = self.hopping_term()
        plaquette_term = self.plaquette_term()
        if plaquette_term is None:
            return hopping_term + mass_term + link_term
        else:
            return hopping_term + mass_term + link_term + plaquette_term
