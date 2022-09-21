from __future__ import annotations

from .second_quantized_op import SecondQuantizedOp
from .fermionic_op import FermionicOp
from .spin_op import SpinOp
from typing import cast

from copy import copy
import numpy as np


class MixedOp(SecondQuantizedOp):
    def _assign_data(self, op):
        if isinstance(op, FermionicOp) or isinstance(op, SpinOp):
            if type(op) in self.ops:
                self.ops[type(op)].append(op)
            else:
                self.ops[type(op)] = [op]
        else:
            raise ValueError(f"Operators of type {type(op)} are not supported.")

    def __init__(
        self,
        data: SecondQuantizedOp
        | list[SecondQuantizedOp]
        | tuple[
            SecondQuantizedOp | list[SecondQuantizedOp],
            float | complex | list[tuple[list[tuple[type(SecondQuantizedOp), int]]]],
        ],
    ):

        # VibrationalOp is currently not supported
        self.ops: dict[type(SecondQuantizedOp), list[SecondQuantizedOp]] = {}
        self.coeffs: list[tuple[list[tuple[type(SecondQuantizedOp), int]], complex]] = []

        if isinstance(data, tuple):
            op_list = data[0]
            if not isinstance(data[1], list):
                coeff = cast("complex", data[1])
        else:
            op_list = data
            coeff = 1

        # first, we create the dictionary of operators
        if isinstance(op_list, list):
            for op in op_list:
                self._assign_data(op)
        else:
            self._assign_data(op_list)

        if not isinstance(data[1], list):
            self.coeffs = [
                ([(FermionicOp, f_index), (SpinOp, s_index)], coeff)
                for f_index in range(len(self.ops[FermionicOp]))
                for s_index in range(len(self.ops[SpinOp]))
            ]
        else:
            self.coeffs = data[1]

    def __repr__(self) -> str:

        repr_fermionic = repr(self.ops[FermionicOp])
        repr_spin = repr(self.ops[SpinOp])

        repr_coeffs = []
        for c in self.coeffs:
            new_coeff = c[1]
            new_ops = [("FermionicOp" if x[0] is FermionicOp else "SpinOp", x[1]) for x in c[0]]
            repr_coeffs.append((new_ops, new_coeff))
        return (
            f"MixedOp( \n "
            f"Operators = ({repr_fermionic}, {repr_spin}\n"
            f"Mix Coefficients = ({repr_coeffs}) \n * )"
        )

    def __len__(self):
        return len(self.ops[FermionicOp]) + len(self.ops[SpinOp])

    def mul(self, other: complex) -> MixedOp:
        if not isinstance(other, (int, float, complex)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'MixedOp' and '{type(other).__name__}'"
            )
        op_list = self.ops[FermionicOp] + self.ops[SpinOp]

        new_coeffs = []
        for c in self.coeffs:
            new_coeff = c[1] * other
            new_ops = [(x[0], x[1]) for x in c[0]]
            new_coeffs.append((new_ops, new_coeff))

        return MixedOp((op_list, new_coeffs))

    def compose(self, other: MixedOp) -> MixedOp:
        raise NotImplementedError

    def add(self, other: FermionicOp | SpinOp | MixedOp) -> MixedOp:

        f_op_list = self.ops[FermionicOp]
        s_op_list = self.ops[SpinOp]
        new_coeffs = copy(self.coeffs)

        if isinstance(other, FermionicOp):
            f_op_list.append(other)
            new_coeff = 1
            new_ops = [(FermionicOp, len(f_op_list) - 1)]
            new_coeffs.append((new_ops, new_coeff))

        elif isinstance(other, SpinOp):
            s_op_list.append(other)
            new_coeff = 1
            new_ops = [(SpinOp, len(s_op_list) - 1)]
            new_coeffs.append((new_ops, new_coeff))

        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: 'MixedOp' and '{type(other).__name__}'"
            )

        op_list = f_op_list + s_op_list
        return MixedOp((op_list, new_coeffs))

    def to_list(
        self,
        display_format: Optional[str] = None,
    ) -> list[tuple[str, complex]]:
        raise NotImplementedError

    def adjoint(self) -> MixedOp:
        raise NotImplementedError

    def simplify(self, atol: Optional[float] = None) -> MixedOp:
        raise NotImplementedError

    @property
    def register_length(self) -> int:
        """Gets the register length."""
        raise NotImplementedError
