"""Deprecated: the first stage 1, built against a finished Kaggle competition's replay
corpus, which is not this project's source anymore. Kept for reference only; see
docs/adr/0001-deprecate-kaggle-source.md for the decision and its consequences.
"""

import warnings

warnings.warn(
    "pipeline.legacy.kaggle is deprecated: the Kaggle replay corpus is no longer this "
    "project's source. See docs/adr/0001-deprecate-kaggle-source.md.",
    DeprecationWarning,
    stacklevel=2,
)
