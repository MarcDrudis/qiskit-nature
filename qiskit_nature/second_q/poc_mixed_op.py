from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.operators import SpinOp
from qiskit_nature.second_q.operators.mixed_op import MixedOp
from qiskit_nature.second_q.mappers.qubit_converter import QubitConverter
from qiskit_nature.second_q.mappers import JordanWignerMapper, LinearMapper
from copy import deepcopy

# REFERENCE QUBIT OPERATORS:
fer_op_1 = FermionicOp("+")
fer_op_2 = 0.5 * FermionicOp("E") + FermionicOp("+")
spin_op_1 = SpinOp("Z")
spin_op_2 = SpinOp("X")


qubit_converter = QubitConverter(mappers=JordanWignerMapper())
fermionic_qubit_op = qubit_converter.convert(fer_op_1)
print("Fermionic qubit op 1: ", fermionic_qubit_op)

qubit_converter = QubitConverter(mappers=JordanWignerMapper())
fermionic_qubit_op = qubit_converter.convert(fer_op_2)
print("Fermionic qubit op 2: ", fermionic_qubit_op)

qubit_converter = QubitConverter(mappers=LinearMapper())
spin_qubit_op = qubit_converter.convert(spin_op_1)
print("Spin qubit op: ", spin_qubit_op)


# # Example 1: Explicit Creation
# mixed_op = MixedOp(([fer_op_1, spin_op_1], 2))
# print(mixed_op)
# qubit_converter = QubitConverter(mappers=[JordanWignerMapper(), LinearMapper()])
# qubit_op = qubit_converter.convert(mixed_op)
# print(qubit_op)
#
# # Example 2: Explicit Creation
# mixed_op = MixedOp(([fer_op_2, spin_op_1], 3))
# print(mixed_op)
# qubit_converter = QubitConverter(mappers=[JordanWignerMapper(), LinearMapper()])
# qubit_op = qubit_converter.convert(mixed_op)
# print(qubit_op)
#
# # Example 3: Implicit Creation
# mixed_op = fer_op_1 @ spin_op_1
# print(mixed_op)
# qubit_converter = QubitConverter(mappers=[JordanWignerMapper(), LinearMapper()])
# qubit_op = qubit_converter.convert(mixed_op)
# print(qubit_op)
#
# # Example 4: Implicit Creation + scalar multiplication
# mixed_op = 3 * (fer_op_1 @ spin_op_1)
# print(mixed_op)
# qubit_converter = QubitConverter(mappers=[JordanWignerMapper(), LinearMapper()])
# qubit_op = qubit_converter.convert(mixed_op)
# print(qubit_op)

# Example 5: Addition
mixed_op = fer_op_1 @ spin_op_1
print("Mixed Op: ", mixed_op)
mixed_op_2 = fer_op_2 @ spin_op_2
print("Mixed Op 2: ", mixed_op_2)
qubit_converter = QubitConverter(mappers=[JordanWignerMapper(), LinearMapper()])
qubit_op = qubit_converter.convert(mixed_op_2)
# print("Qubit Op 2: ", qubit_op)

mixed_op_3 = mixed_op + fer_op_2
print("Mixed Op 3: ", mixed_op_3)
qubit_converter = QubitConverter(mappers=[JordanWignerMapper(), LinearMapper()])
qubit_op = qubit_converter.convert(mixed_op_3)
# print("Qubit Op 3: ", qubit_op)

mixed_op_4 = mixed_op + mixed_op_3
print("Mixed Op 4: ", mixed_op_4)
qubit_converter = QubitConverter(mappers=[JordanWignerMapper(), LinearMapper()])
qubit_op = qubit_converter.convert(mixed_op_4)
print("Qubit Op 4: ", qubit_op)
