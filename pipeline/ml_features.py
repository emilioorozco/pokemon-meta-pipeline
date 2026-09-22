"""The feature contract: what the model is given, and how a row becomes numbers.

This module exists because two commands have to agree about it and they run at
different times on different machines. `pipeline.train` builds the design
matrix from a warehouse table; `pipeline.serve` builds one row of it from a
JSON body months later. If the two kept their own lists, the day someone adds a
feature to the trainer is the day the service starts sending the model a matrix
with the columns in the wrong order, which is not an error anywhere: LightGBM
takes the numbers it is given and returns a confident wrong answer.

So the list lives here once. `train.py` logs it as `features.json` and the
request model in `serve.py` is checked against it by a test, which means a
feature added in one place fails the suite rather than the prediction.

`design_matrix` is here for the same reason. The encoding is part of the
contract: archetypes are integers fitted on the training split, and an
archetype nobody trained on is -1, which LightGBM reads as a missing category.
Serving does not reimplement that, it calls this.
"""

from typing import Final

import pandas as pd

# The columns the model is given, in this order. `features.json` is logged from
# this tuple and the serving request model mirrors it, so the artifact, the
# design matrix and the HTTP body cannot disagree.
MODEL_FEATURES: Final[tuple[str, ...]] = (
    "turn_number",
    "went_first",
    "archetype_key",
    "opponent_archetype_key",
    "prizes_taken_self",
    "prizes_taken_opp",
    "prize_diff",
    "knockouts_self",
    "knockouts_opp",
    "cards_drawn_self",
    "energy_attached_self",
    "pokemon_played_self",
    "trainers_played_self",
    "evolutions_self",
    "attacks_self",
    "turns_played_self",
)
CATEGORICAL: Final[tuple[str, ...]] = ("archetype_key", "opponent_archetype_key")
LABEL: Final = "won"
# What an archetype the training split never held encodes to. LightGBM treats a
# negative category as missing, which is the truth about it: the model has
# never seen that deck.
UNSEEN_CATEGORY: Final = -1

# The type of the code map both the trainer writes and the service reads back:
# one mapping per categorical column, archetype key to integer code.
ArchetypeCodes = dict[str, dict[str, int]]


def design_matrix(frame: pd.DataFrame, codes: ArchetypeCodes) -> pd.DataFrame:
    """The feature columns as numbers, in `MODEL_FEATURES` order.

    Everything is numeric by the time it leaves here: booleans become 0 and 1,
    archetypes become their training-split code, and an archetype the training
    split never held becomes `UNSEEN_CATEGORY`.
    """
    matrix = pd.DataFrame(index=frame.index)
    for column in MODEL_FEATURES:
        if column in codes:
            mapping = codes[column]
            matrix[column] = [mapping.get(value, UNSEEN_CATEGORY) for value in frame[column]]
            matrix[column] = matrix[column].astype("int32")
        else:
            matrix[column] = frame[column].astype("int32")
    return matrix
