from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit.circuit.controlflow import IfElseOp

from typing import Tuple


def decompose(circuit: QuantumCircuit) -> Tuple[QuantumCircuit]:
    """Decompose a branched qiskit circuit into its parts.

    Naive break-down of a qiskit circuit into parts leading up to and
    following an if_else with 2 branches.

    Parameters
    ----------
      circuit: QuantumCircuit
        quantum circuit to sample from

    Returns
    -------
      tuple of base, zero-branch, and one-branch circuits
    """

    qubits = circuit.qubits
    clbits = circuit.clbits

    base_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)

    for instr in circuit.data:
        if isinstance(instr.operation, IfElseOp):
          # TODO: the following may be too simplistic, but it's all we're using atm.
            assert instr.operation.condition[1] in (0, 1)
            if instr.operation.condition[1] == 0:     # zero is True branch
                iZ = 0; iO = 1
            else:                                     # one is True branch
                iZ = 1; iO = 0

            branches = instr.operation.params
            zero_circuit = branches[iZ] and branches[iZ].copy() or None
            one_circuit  = branches[iO] and branches[iO].copy() or None

        for qubit in instr.qubits:
            index = qubits.index(qubit)

        if instr.operation.name != "measure":
            base_circuit.append(CircuitInstruction(
                instr.operation,
                [base_circuit.qubits[qubits.index(q)] for q in instr.qubits],
                [base_circuit.clbits[clbits.index(c)] for c in instr.clbits])
            )

    return base_circuit, zero_circuit, one_circuit

