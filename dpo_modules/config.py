"""
Configuration for GuessDPOTrainer.

Fields
------
beta                float   0.1   DPO temperature (KL penalty coefficient).
rl_coeff            float   0.05  Weight of REINFORCE loss relative to DPO loss.
                                  Set to 0 to disable the judge step entirely.
hardness_weighting  bool    True  Multiply DPO loss by 2 for pairs the model
                                  guessed wrong; 1 otherwise.  Surfaces hard
                                  examples without changing the sign of the loss.
judge_template      str     see   Prompt template for the A/B judgment task.
                            below Keys: {prompt}, {a}, {b}.
judge_max_length    int     1024  Max tokens for judge prompts (truncated if longer).
implicit_mode       bool    False Use log-prob ratio as the implicit "guess" instead
                                  of a separate generation step.  No extra forward
                                  pass; cheaper but less expressive.
"""
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Default judge template
# ---------------------------------------------------------------------------

_DEFAULT_JUDGE_TEMPLATE = (
    "Below are two responses to the following prompt.\n\n"
    "Prompt:\n{prompt}\n\n"
    "Response A:\n{a}\n\n"
    "Response B:\n{b}\n\n"
    "Which response is better? Reply with a single letter.\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuessDPOConfig:
    beta: float = 0.1
    rl_coeff: float = 0.05
    hardness_weighting: bool = True
    judge_template: str = field(default=_DEFAULT_JUDGE_TEMPLATE)
    judge_max_length: int = 1024
    implicit_mode: bool = False
