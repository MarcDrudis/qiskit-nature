from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.operators import SpinOp
from qiskit_nature.second_q.operators.mixed_op import MixedOp

fer_op_1 = FermionicOp("+", display_format="dense")
fer_op_2 = FermionicOp("+") + (0.5 * FermionicOp("E"))
spin_op_1 = SpinOp("Z")


print("fermionic op: ", fer_op_2)
print("spin op: ", spin_op_1)
print(" --------- ")
mo = MixedOp(([fer_op_2, spin_op_1], 2))
print(" --------- ")
print(mo)

from qiskit_nature.second_q.mappers.qubit_converter import QubitConverter
from qiskit_nature.second_q.mappers import JordanWignerMapper, LinearMapper

qubit_converter = QubitConverter(mappers=[JordanWignerMapper(), LinearMapper()])
qubit_op = qubit_converter.convert(mo)

mo3 = 2 * mo
print("el 3: ", mo3)

mo4 = mo + fer_op_2 + spin_op_1
print("el 4: ", mo4)

qubit_op = qubit_converter.convert(mo4)
print("qubit op: ", qubit_op)

qubit_converter = QubitConverter(mappers=JordanWignerMapper())
fermionic_qubit_op = qubit_converter.convert(fer_op_2)
print("Fermionic qubit op: ", fermionic_qubit_op)

qubit_converter = QubitConverter(mappers=LinearMapper())
spin_qubit_op = qubit_converter.convert(spin_op_1)
print("Spin qubit op: ", spin_qubit_op)
