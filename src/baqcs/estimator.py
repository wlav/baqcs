class Estimator:
    def __init__(self):
        self.cache = dict()

    def reset(self):
        self.cache.clear()

    def _estimate_p_single(outcome, initial, branches, system_qubits, ancilla_qubits):
        """
        Estimate measurement probabilities for system qubits given ancilla
        measurement outcome. This implementation uses the full circuit info.

        Args:
            outcome: 0 or 1 - the measured ancilla value
            initial: QuantumCircuit with operations before ancilla measurement
            branches: Tuple of QuantumCircuit branches indexed with ancilla measurement
            system_qubits: List of indices for system qubits
            ancilla_qubits: List of indices for ancilla qubits

        Returns:
            torch.Tensor: Probability table P(system|ancilla)
        """

        try:
            system_probs = self.cache[outcome]
        except KeyError:
            pass

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

        n_system = len(system_qubits)
        system_dim = 2**n_system

      # for single ancilla qubit, create mask to check if it matches outcome
        assert len(ancilla_qubits) == 1
        ancilla_qubit = ancilla_qubits[0]  # Assumes single ancilla

      # iterate over all possible states to marginalize the system probabilites
        system_probs = torch.zeros(system_dim)

        for state_idx, prob in enumerate(probs):
          # add probabilities if ancilla matches the measurement outcome
            ancilla_bit = (state_idx >> ancilla_qubit) & 1
            if ancilla_bit == outcome:
              # extract system state index using actual qubit positions
                system_idx = 0
                for q in system_qubits:
                    bit = (state_idx >> q) & 1
                    system_idx += bit * 2**q

                system_probs[system_idx] += prob

      # normalize (since probabilities were post-selected) after adding
      # a small fudge to help with numerical stability in case noisy
      # data generate outcomes that aren't technically possible
        system_probs = system_probs + 1e-8
        system_probs /= system_probs.sum()

        self.cache[outcome] = system_probs
        return system_probs


    def estimate_p(outcome, initial, branches, system_qubits, ancilla_qubits):
        """
        Compute measurement probabilities, handling both single values and
        enumerated tensors.

        Args:
            outcome: 0 or 1 - the measured ancilla value
            initial: QuantumCircuit with operations before ancilla measurement
            branches: Tuple of QuantumCircuit branches indexed with ancilla measurement
            system_qubits: List of indices for system qubits
            ancilla_qubits: List of indices for ancilla qubits

        Returns:
            torch.Tensor: Probability table P(system|ancilla)
        """

      # special case if outcome is a tensor from enumeration
        if torch.is_tensor(outcome) and outcome.dim() > 0:
          # compute the probabilities for each enumerated value
            all_probs = torch.stack([
                estimate_p_single(o, initial, branches, system_qubits, ancilla_qubits)
                for o in outcome
            ])

            return all_probs

      # else single value
        return estimate_p_single(int(outcome), initial, branches, system_qubits, ancilla_qubits)

