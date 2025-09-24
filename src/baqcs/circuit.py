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
          # TODO: AFAICT, it can not actually be derived from the instruction
          # what the comparison is against and hence which branch is what, so
          # simply hard-wire the branch collection here
            zero_circuit = instr.operation.params[1].copy()
            one_circuit  = instr.operation.params[0].copy()
            break

        for qubit in instr.qubits:
            index = qubits.index(qubit)

        if instr.operation.name != "measure":
            base_circuit.append(CircuitInstruction(
                instr.operation,
                [base_circuit.qubits[qubits.index(q)] for q in instr.qubits],
                [base_circuit.clbits[clbits.index(c)] for c in instr.clbits])
            )

    return base_circuit, zero_circuit, one_circuit

