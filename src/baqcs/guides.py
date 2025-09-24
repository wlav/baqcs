import pyro
import torch

import pyro.distributions as dist
import torch.nn.functional as F


def logit_guide(observations=None):
    """
    Variational guide for inference.
    """

    if observations is not None:
        n_obs = len(observations)
    else:
        n_obs = 1

    # Global parameters for ancilla distribution
    ancilla_logits = pyro.param("ancilla_logits", torch.zeros(2))
    ancilla_probs = F.softmax(ancilla_logits, dim=0)

    with pyro.plate("data", n_obs):
        # Each observation has its own ancilla
        ancilla = pyro.sample(
            "ancilla",
            dist.Categorical(ancilla_probs),
            infer={"enumerate": "parallel"}
        )

    return ancilla_probs  # Return the learned distribution

  # random initialization around 50/50 (TODO: allow initialization
  # from a seed and make the spread configurable)
    init_logits = 0.5 + torch.randn(2) * 0.05

    ancilla_logits = pyro.param("ancilla_logits", init_logits)
    ancilla = pyro.sample(
        "ancilla",
        dist.Categorical(logits=ancilla_logits),
        infer={"enumerate": "parallel"}
    )

    return ancilla
