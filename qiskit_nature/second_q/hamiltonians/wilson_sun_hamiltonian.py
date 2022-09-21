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
    """The Wilson Model model."""

    def __init__(
        self,
        lattice: HyperCubicLattice,
        a: complex,
        r: complex,
        q: float,
        mass: float,
        e_value: float,
        lmbda: float,
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
        self._a = a
        self._q = q
        self._r = r
        self._lambda = lmbda
        self.e = e_value
        self._mass = mass
        self._d = lattice.dim
        self.representation = representation
        self.ncomponents = int(2 ** ((self._d + 1) // 2))
        self._fermionic_spinor = FermionicSpinor(ncomponents=self.ncomponents, lattice=lattice)
        self._QLM_spin = QLM(
            spin=spin,
            lattice=lattice,
            e_value=e_value,
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
                self._fermionic_spinor.spinor_product(site, site, self.representation[0])
            )
        return (self._mass + self._r * self._d / self._a) * sum(mass_terms)

    def hopping_term(self) -> MixedOp:
        """Creates the hopping term in the hamiltonian.
        Note that this only creates the first term and we still need to add the conjugate transpose.
        """
        hopping_terms = []

        tensor_size = self.representation[0].shape[0]
        for edge_index, (node_a, node_b) in enumerate(self.lattice.graph.edge_list()):
            k = self.lattice.direction((node_a, node_b))
            fermionic_tensor = self.representation[0] @ (
                self._r * np.eye(tensor_size) + 1.0j * self.representation[np.abs(k)]
            )
            fermionic_part = self._fermionic_spinor.spinor_product(node_a, node_b, fermionic_tensor)

            bosonic_part = self._QLM_spin.operatorU(edge_index=edge_index)

            mix_op = MixedOp(([fermionic_part, bosonic_part], 1 / (2 * self._a)))
            mix_op_dag = MixedOp(
                ([fermionic_part.adjoint(), bosonic_part.adjoint()], 1 / (2 * self._a))
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
            plaquette_terms.append(self._QLM_spin.operator_plaquette(*plaquette))
            plaquette_terms.append(self._QLM_spin.operator_plaquette(*plaquette).adjoint())

        return (-1) / (4 * self.e) * sum(plaquette_terms)

    def link_term(self) -> SpinOp:
        link_terms = []
        for edge_index, (node_a, node_b) in enumerate(self.lattice.graph.edge_list()):
            op_E = self._QLM_spin.operatorE(edge_index=edge_index)
            op_E_2 = self._QLM_spin.operatorE_2(edge_index=edge_index)
            idnty = self._QLM_spin.idnty()
            k = self.lattice.direction((node_a, node_b))
            field = self._QLM_spin.electric_field[k - 1]
            link_terms.append(op_E_2 + 2 * field * op_E + field**2 * idnty)

        return self.e**2 / 2 * sum(link_terms)

    def gauss_operators(self) -> List[MixedOp]:
        charge_offset = -self.e
        gauss_terms = []
        for site in self.lattice.node_indexes:
            gauss_term = MixedOp(([self._fermionic_spinor.idnty(),self._QLM_spin.operator_divergence(site)],1))
            gauss_term += -self._q * self._fermionic_spinor.spinor_product(site, site, None)
            gauss_terms.append(gauss_term)
        return gauss_terms

    def mock_qubit_operator(self):
        return sum(self.mock_qubit_parts())

    def second_q_ops(self, display_format: str = "sparse"):
        """Return the Hamiltonian of the Wilson Model in terms of `MixedOp.

        Args:
            display_format: If sparse, the label is represented sparsely during output.
                If dense, the label is represented densely during output. Defaults to "sparse".

        Returns:
            MixedOp: The Hamiltonian of the Wilson Model.
        """

        mass_term = self.mass_term()
        link_term = self.link_term()
        hopping_term = self.hopping_term()
        # plaquette_term = self.plaquette_term()
        return hopping_term + mass_term + link_term


















    # def mock_qubit_parts(self):
    #     #Pure mass term
    #     mass_op = self.mass_term()

    #     #Pure link_plaquette term
    #     link_plaquette_op = self.link_term()
    #     if self._d > 1:
    #         link_plaquette_op += self.plaquette_term()

    #     #Fermionic Spin and converter
    #     fermionic_converter = QubitConverter(JordanWignerMapper())
    #     fermionic_idnty = FermionicOp.one(register_length=mass_op.register_length)
    #     fermionic_idnty_qubits = fermionic_converter.convert(fermionic_idnty)

    #     #Spin identity and converter
    #     spin_converter = QubitConverter(LogarithmicMapper())
    #     spin_idnty = SpinOp(
    #         "", spin=link_plaquette_op.spin, register_length=link_plaquette_op.register_length
    #     )
    #     spin_idnty_qubits = spin_converter.convert(spin_idnty)

    #     #Tensored link_plaquette term
    #     link_plaquette_qubits = spin_converter.convert(link_plaquette_op)
    #     link_plaquette_term = fermionic_idnty_qubits ^ link_plaquette_qubits

    #     #Tensored mass term
    #     mass_qubits = fermionic_converter.convert(mass_op)
    #     mass_term = mass_qubits ^ spin_idnty_qubits


    #     #hopping_term
    #     hopping_term = None
    #     for coeff, ferm, spi in self.hopping_term():
    #         if hopping_term is None:
    #             hopping_term = coeff * (
    #             fermionic_converter.convert(ferm) ^ spin_converter.convert(spi)
    #             )
    #         else:
    #             hopping_term += coeff * (
    #                 fermionic_converter.convert(ferm) ^ spin_converter.convert(spi)
    #             )

    #     # Regulator term
    #     regulator_term = None
    #     gauss_operators,charge_offset = self.gauss_operators()
    #     for divergence,charge in gauss_operators:
    #         if divergence is None:
    #             div_op = spin_idnty_qubits
    #         elif divergence == 0:
    #             continue
    #         else:
    #             div_op = spin_converter.convert(divergence)
    #         ch_op = fermionic_converter.convert(charge)
    #         gauss_op = (fermionic_idnty_qubits^div_op) #+ (ch_op^spin_idnty_qubits)
    #         gauss_op -= charge_offset * (fermionic_idnty_qubits^spin_idnty_qubits)
    #         if regulator_term is None:
    #             regulator_term = self._lambda * (gauss_op@gauss_op)
    #         else:
    #             regulator_term += self._lambda * (gauss_op@gauss_op)


    #     return  (self._a**self._d * hopping_term, self._a**self._d * mass_term, self._a**self._d * link_plaquette_term, regulator_term)
