from collections import defaultdict

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


class Sampler:
    def __init__(self):
        self.simulator = AerSimulator()

    def sample(self, circuit: QuantumCircuit, shots: int=8192):
        """Sample measurements from the provided quantum circuit.

        Run the provided circuit on the Qiskit Aer simulutor and sample it
        for measurement counts split by system and ancilla qubits. The
        circuit may be partial, in which case final measurements are added
        for running the simulation.

        Parameters
        ----------
          circuit: QuantumCircuit
            quantum circuit to sample from
          shots: int
            number of samples to compute

        Returns
        -------
          tuple of system, ancilla, and total counts
        """

        assert circuit.num_ancillas and "ancilla qubits expected, but none found"

        ancillas = set(circuit.ancillas)
        sys_qubits = set(circuit.qubits) - ancillas

        Ns = len(sys_qubits)
        Na = circuit.num_ancillas

      # add system qubit measurements if needed
        targets = set()
        for instr in reversed(circuit.data):
            if instr.operation.name == "measure":
                targets.add(instr.qubits)
            else:
                break

        if not sys_qubits.issubset(targets):
            sampler_qc = circuit.copy()
            sampler_qc.barrier()
            sampler_qc.measure(sys_qubits, circuit.clbits[:Ns])
        else:
            sampler_qc = circuit

      # simulate
        sim_qc = transpile(sampler_qc, self.simulator)
        result = self.simulator.run(sim_qc, shots=shots).result()
        counts = result.get_counts(sim_qc)

      # split results along system and ancilla qubits
        sys_counts, anc_counts = defaultdict(int), defaultdict(int)
        for result, count in counts.items():
          # bitstrings are ordered ancillas, system qubits
            if len(result) == Ns+Na:
                sys_counts[result[Na:]] += count
                anc_counts[result[:Na]] += count
            else:
                registers = result.split()
                assert len(registers) == 2
                assert len(registers[0]) == Na
                assert len(registers[1]) == Ns
                sys_counts[registers[1]] += count
                anc_counts[registers[0]] += count

        return sys_counts, anc_counts, counts
