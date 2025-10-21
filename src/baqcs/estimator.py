import collections.abc
import torch

from qiskit.quantum_info import Statevector


class Estimator:
    def estimate_p_single(self, outcome, initial, branches):
        """
        Estimate measurement probabilities for system qubits given ancilla
        measurement outcome. This implementation uses the full circuit info.

        Args:
            outcome: 0 or 1 - the measured ancilla value
            initial: QuantumCircuit with operations before ancilla measurement
            branches: Tuple of QuantumCircuit branches indexed with ancilla measurement

        Returns:
            torch.Tensor: Probability table P(system|ancilla)
        """

      # get the state after initial operations from the initial plus
      # chosen branch based on the ancilla measuement to calculate the
      # expected system qubits out comes
      #  TODO:
      #     - get the projection from calculated probabilities only
      #     - allow inclusion of error estimates
      #     - export information to use as a building block

      # construct full circuit with the branch taken based on ancilla
      # measurement outcome
        branched_circuit = initial.copy()
        branch = branches[0] if outcome == 0 else branches[1]
        branched_circuit.compose(branch, inplace=True)

        probs = Statevector.from_instruction(branched_circuit).probabilities()

        n_system = len(branched_circuit.qubits) - len(branched_circuit.ancillas)
        system_dim = 2**n_system

      # iterate over all possible states to marginalize the system probabilites
        system_probs = torch.zeros(system_dim)

        for state_idx, prob in enumerate(probs):
          # add probabilities (ancilla matches by definition b/c of the chosen
          # branch); system state index assumes system qubits come first
            system_idx = 0
            for q in range(n_system):
                bit = (state_idx >> q) & 1
                system_idx += bit * 2**q

            system_probs[system_idx] += prob

        return system_probs

