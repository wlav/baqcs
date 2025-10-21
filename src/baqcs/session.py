import collections.abc
import numpy as np
import torch
import pyro

from pyro.optim import Adam
from pyro.infer import (
    SVI, TraceEnum_ELBO,
    MCMC, NUTS,
)
import torch.nn.functional as F

from qiskit.quantum_info import Statevector

from typing import Dict, List


class Session:
    def __init__(self):

      # reset pyro session
        pyro.enable_validation(True)
        pyro.clear_param_store()

      # memoization for performance
        self.estimate_p_cache = dict()

    def _encode_data(self, counts: Dict[str, int]) -> torch.tensor:
        """Experimental data (IBM format) to model encoding"""

        meas = []
        for bitstring, count in counts.items():
            bits = bitstring[::-1]     # expected in IBM ordering

          # measurements are encoded as: 00->0, 01->1, 10->2, 11->3 etc.
            m = int(bits, 2)
            meas.extend([m] * count)

      # randomize the data order (as given it's sorted as it came from the
      # histogram, but actual measurements will have been random)
        data = torch.tensor(meas)
        indexes = torch.randperm(data.shape[0])
        observations = data[indexes]

        return observations

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

        try:
            system_probs = self.estimate_p_cache[outcome]
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

        n_system = len(branched_circuit.qubits)
        system_dim = 2**n_system

      # for single ancilla qubit, create mask to check if it matches outcome
        assert len(branched_circuit.ancillas) == 1
        ancilla_qubit = n_system  # assumes single ancilla following system qubits

      # iterate over all possible states to marginalize the system probabilites
        system_probs = torch.zeros(system_dim)

        for state_idx, prob in enumerate(probs):
          # add probabilities if ancilla matches the measurement outcome
            ancilla_bit = (state_idx >> ancilla_qubit) & 1
            if ancilla_bit == outcome:
              # extract system state index (assumes system qubits come first)
                system_idx = 0
                for q in range(n_system):
                    bit = (state_idx >> q) & 1
                    system_idx += bit * 2**q

                system_probs[system_idx] += prob

      # normalize (since probabilities were post-selected) after adding
      # a small fudge to help with numerical stability in case noisy
      # data generate outcomes that aren't technically possible
        system_probs = system_probs + 1e-8
        system_probs /= system_probs.sum()

        self.estimate_p_cache[outcome] = system_probs
        return system_probs


    def estimate_p(self, outcome, initial, branches):
        """
        Compute measurement probabilities, handling both single values and
        enumerated tensors.

        See estimate_p_single for detailed description of the arguments.
        """

      # multiple outcomes requested?
        if torch.is_tensor(outcome) and outcome.dim() > 0 or \
                isinstance(outcome, collections.abc.Iterable):
          # compute the probabilities for each enumerated value
            all_probs = torch.stack([
                self.estimate_p_single(int(o), initial, branches) for o in outcome
            ])

            return all_probs

      # single value case
        return self.estimate_p_single(int(outcome), initial, branches, system_qubits, ancilla_qubits)

    def svi(self, sys_qubits, model, guide, observations, nsteps = 100, seed: int | None = None):
      # setup SVI, using an Adam optimizer with momentum and use more particles
      # to improve gradient estimates (TODO: picked a large number b/c it worked,
      # but it's obviously very slow; need tuning)
        pyro.clear_param_store()
        self.estimate_p_cache.clear()

        Ns = len(sys_qubits)

      # optional: fixed seeds for reproducibility
        if seed is not None:
            pyro.set_rng_seed(seed)
            np.random.seed(42)

        optimizer = pyro.optim.Adam({"lr": 0.01, "betas": (0.95, 0.999)})
        elbo = TraceEnum_ELBO(max_plate_nesting=1, num_particles=50)
        svi = SVI(model, guide, optimizer, loss=elbo)

      # training
        losses = []
        param_history = []

      # run optimization
        for step in range(nsteps):
            loss = svi.step(observations)
            losses.append(loss)

            with torch.no_grad():
                logits = pyro.param("ancilla_logits").detach()
                probs = F.softmax(logits, dim=0)
                param_history.append(probs.clone())

            if step % 200 == 0:
                print(f"Step {step:4d}, Loss: {loss:8.4f}, "
                      f"P(ancilla=1): {probs[1]:.4f}")

      # store and report final result
        final_logits = pyro.param("ancilla_logits").detach()
        final_probs = F.softmax(final_logits, dim=0)

        return final_probs

    def run_mcmc_inference(self, model, observations, cpt, num_samples=2000, warmup_steps=500):
        """
        Run MCMC inference to estimate ancilla measurement distribution.
        """

        if not isinstance(observations, torch.Tensor):
            observations = self._encode_data(observations)

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
        mcmc.run(observations, cpt)

        return mcmc.get_samples()

