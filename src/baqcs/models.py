import torch
import pyro

from pyro.distributions import Categorical, Beta, Bernoulli
from pyro.infer import config_enumerate


@config_enumerate
def generic_ancilla_model_with_circuit(
        probability_table, observations=None):
    """
    Generalized probabilistic model for circuits with ancilla measurements.

    Args:
        probability_table: System qubits probability table per ancilla
        observations: Observed system measurements
    """

  # TODO: this code assumes qubits (enumeration 0, 1).
    if observations is not None:
        n_obs = len(observations)
    else:
        n_obs = 1

  # Prior for ancilla (uniform or from circuit)
    ancilla_prior = torch.ones(2) / 2  # Uniform prior

    with pyro.plate("data", n_obs):
      # Each observation has its own (latent) ancilla state
      # The enumerate decorator will enumerate over this variable
        ancilla = pyro.sample("ancilla", Categorical(ancilla_prior),
                            infer={"enumerate": "parallel"})

      # Select the appropriate probabilities based on ancilla values
      # When enumerated, ancilla will have an extra dimension
        if ancilla.dim() == 1:  # Not enumerated (shouldn't happen with decorator)
            system_probs = probability_table[ancilla.long()]
        else:  # Enumerated case - has extra enumeration dimension
          # Ancilla shape: [2, n_obs] where 2 is enumeration dimension
          # We need to gather the right probabilities
            system_probs = probability_table[ancilla.long()]

      # Observe system measurements
        system = pyro.sample("system", Categorical(system_probs), obs=observations)

    return ancilla


def binomial_ancilla_model(observed_system_data, cpt):
    """
    Probabilistic model for inferring ancilla measurements.

    Args:
        observed_system_data: Observed system qubit measurements
        cpt: Conditional probability table P(system|ancilla)

    """
    n_samples = len(observed_system_data)

  # Prior for the probability of ancilla being 1 (binomial parameter)
    p_ancilla_one = pyro.sample("p_ancilla_one", Beta(1, 1))

  # For each observation
    with pyro.plate("data", n_samples):
        # Sample ancilla state from binomial distribution
        ancilla = pyro.sample("ancilla", Bernoulli(p_ancilla_one))

      # Get the conditional probabilities for this ancilla state
      # Convert to long tensor for indexing
        ancilla_long = ancilla.long()
        probs = cpt[ancilla_long, :]

      # Observe system measurement given ancilla state
        pyro.sample("system", Categorical(probs),
                    obs=observed_system_data)

