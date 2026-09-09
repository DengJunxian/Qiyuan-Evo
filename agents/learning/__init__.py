"""Agent learning facade.

The default path is deterministic and heuristic. Larger RL or LLM-backed
training systems can plug into these interfaces without becoming required for
normal simulation runs.
"""

from agents.learning.imitation_dataset import ImitationDataset, ImitationSample
from agents.learning.marl_env import MARLEnvironment, MARLStepResult
from agents.learning.persona_prior import PersonaPrior, kl_to_prior
from agents.learning.policy_heads import HeuristicPolicyHead, PolicyHeadOutput, select_policy_head
from agents.learning.regime_router import RegimeRoute, RegimeRouter

__all__ = [
    "HeuristicPolicyHead",
    "ImitationDataset",
    "ImitationSample",
    "MARLEnvironment",
    "MARLStepResult",
    "PersonaPrior",
    "PolicyHeadOutput",
    "RegimeRoute",
    "RegimeRouter",
    "kl_to_prior",
    "select_policy_head",
]
