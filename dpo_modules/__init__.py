"""
dpo_modules: DPO with online judge self-play (GuessDPO).

GuessDPOTrainer adds an explicit preference-judgment step to standard DPO:

  1. For each (prompt, chosen, rejected) pair the *current* model generates an
     explicit A/B preference over the two responses (the "guess").
  2. A REINFORCE signal rewards correct guesses (+1) and penalises wrong ones (-1).
  3. The DPO loss is optionally upweighted 2× for pairs the model guessed wrong,
     surfacing hard examples automatically.

This creates a self-improving loop: a model that generates better responses
should also judge them more reliably, and examples it misjudges receive stronger
gradient corrections.
"""

from .config import GuessDPOConfig
from .guess_dpo_trainer import GuessDPOTrainer

__all__ = ["GuessDPOConfig", "GuessDPOTrainer"]
