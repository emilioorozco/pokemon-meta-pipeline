"""The application's analysis facts, and the deterministic check on the numbers.

The application computes a sidecar for every game it parses: a few dozen small
numeric statements about that one game, "your first attack was on turn 3",
"you took no prize", each one a pure function of the member's own log. PLA-188
hands the player-perspective half of that sidecar to the agent, so that a
question like "which turns did I not attack" has something to be answered out
of. Nothing in the warehouse can answer it: every mart is an aggregate across
games and no game-level table is on the allowlist.

A fact is three fields and no more. `id` is a stable key the application owns,
`text` is one plain sentence containing its numbers, and `values` are those
numbers as the application computed them. The sentence is what the model
reads; the values are what this module checks the answer against, which is why
both are sent rather than either alone. A sentence with no values beside it
could only be checked by parsing it, and parsing somebody else's prose is how
a check starts disagreeing with the thing it checks.

**The facts are not trusted.** They are built by the application, which is the
same third kind of input the page context is, and rule 9 of the prompt covers
them for the same reason: a sentence inside the element that reads like an
order is text on a page. Our own delimiters come out of the text before it is
placed, exactly as they come out of the question and the context
(docs/agent-safety.md).

**The numeric check is a string search and not a judge.** After the answer
comes back, every number-shaped token in the prose is looked up in the rows
the queries returned, the fields of the cards that were read, the `values` of
the facts, and a four-entry allowlist. A number found nowhere is reported on
the response as `unverified_numbers`, and that is the whole of it: the answer
still returns, nothing is refused, and no second model is asked for an
opinion. What this catches is the failure the facts make likelier, which is a
model that has a sentence full of turn numbers in front of it and writes one
that was never in it. What it cannot catch is a number that is real and
irrelevant, or a sentence that is wrong about numbers it quotes correctly;
docs/agent-safety.md says so out loud.

**An earlier answer is a fourth place a number can come from, and it is
reported apart.** Since PLA-204 a follow-up carries the last few turns of the
conversation back with it, so the model has its own previous answers in front
of it. A number out of one of those is neither an invention nor a finding: it
is this agent quoting itself about rows it has not read again, which is what
rule 11 of the prompt is about. So the check searches the assistant's turns
last, and what it finds only there comes back as `from_history` rather than
inside `unverified_numbers`. Two counts, because "the model made a number up"
and "the model repeated its own number" are two different things to be
looking at.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from pipeline.prompts import clean_fact_text

# The contract the application codes against, and the ceilings this service
# enforces. Sixty facts of two hundred characters is twelve thousand
# characters at the very worst, which is three times a context and still a
# small share of a question's input; in practice a game produces a dozen.
MAX_FACTS: Final = 60
MAX_FACT_ID_CHARS: Final = 64
MAX_FACT_TEXT_CHARS: Final = 200

# The numbers an answer may write without anything to trace them to. Nought,
# one and two are the counting words a correct sentence reaches for whatever
# the data says ("one of the two sides", "no prizes"), and a hundred is the
# denominator of every percentage. Anything else has to come from a row, a
# card or a fact.
ALLOWED_NUMBERS: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 100.0)

# The fact whose value is the length of the game, which the contract asks to
# be allowed beside the four constants above. It needs no clause of its own:
# the application sends it as a fact, and every fact's values are allowed, so
# an answer that says "on 4 of your 9 turns" has its 9 accounted for. The name
# is here so a reader can find the fact the sentence is talking about, and so
# that a rename in the application is one grep.
TURN_COUNT_FACT_ID: Final = "turn_count"

# How close two numbers have to be to be the same number. Floats that came out
# of JSON and floats that came out of DuckDB agree to far better than this.
TOLERANCE: Final = 1e-9

# The scan, in two halves, and deliberately crude before it is strict.
#
# `_NUMBER_TOKEN` takes a digit that does not continue a word and everything
# number-like after it, so `2026-09-14`, `1.2.3` and `9.` all arrive whole
# rather than as the pieces a tighter pattern would cut them into.
# `_PLAIN_NUMBER` then decides: a token is a number only if, with sentence
# punctuation taken off the end, what is left is an integer, an integer with
# thousands separators, a decimal, or any of those with a per cent sign.
#
# Everything else is dropped rather than split, which is the conservative
# direction. A date, a version, a hyphenated record (`6-2`) and an identifier
# with a digit in it are all left alone, so this under-reports rather than
# filling a member's receipt with numbers nobody claimed. Turn words spelled
# out ("turn nine") are not numbers and never were.
_NUMBER_TOKEN: Final = re.compile(r"(?<![\w$])\d[\d,.%-]*")
_PLAIN_NUMBER: Final = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?%?\Z")
# The punctuation a sentence leaves on the end of a number.
_TRAILING: Final = ".,;:)]}"

# How an answer cites a fact by its number: "fact 2", "facts 2 and 3", "fact
# #2". The numbered list is one to ten items long, so a bare "2" is a number
# and not a citation; the word has to be there.
_FACT_REFERENCE: Final = re.compile(r"\bfacts?\s*#?\s*(\d+)", re.IGNORECASE)


class FactError(ValueError):
    """A fact the application sent is not one this service will place."""


@dataclass(frozen=True)
class Fact:
    """One statement the application computed about the game on the member's screen.

    Frozen and three fields, because this is a wire contract before it is a
    type: the application builds it, `POST /ask` validates it, the prompt
    renders `text` and the numeric check reads `values`.
    """

    id: str
    text: str
    values: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "values": list(self.values)}


@dataclass(frozen=True)
class FactEvidence:
    """One fact as the answer's receipt reports it: what it said, and whether it was used.

    `cited` is the half the application draws on. A fact that was placed and
    never used is not a failure, but ten of them on every question is a sign
    the facts are the wrong facts, and that is a thing to be able to count.
    """

    id: str
    text: str
    cited: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "cited": self.cited}


@dataclass(frozen=True)
class NumberCheck:
    """What the numeric check found: the numbers nowhere, and what it searched.

    `known` is here for the tests and for a reader debugging a false report,
    and nothing serialises it: the response carries the two lists of strings
    and their counts, which is all an application can act on.
    """

    unverified: tuple[str, ...] = ()
    known: frozenset[float] = field(default_factory=frozenset)
    # Numbers this run cannot account for that an earlier answer in the
    # conversation stated. Reported apart from `unverified` because the two
    # mean different things: one is a number nobody wrote and the other is a
    # number this agent wrote before, out of rows it has not read again
    # (rule 11 of the prompt, docs/agent-service.md).
    from_history: tuple[str, ...] = ()


def clean_facts(facts: Sequence[Fact] | None) -> tuple[Fact, ...]:
    """The facts as they will be placed: delimiters out, empties dropped, capped.

    Every fact goes through `clean_fact_text`, which takes our own element tags
    out of the sentence and puts it on one line, for the reason `wrap_question`
    strips the question: a fact containing `</facts>` would otherwise close the
    list early and put the rest of itself where the model has been told the
    project's own words are.

    A fact whose sentence is empty once that is done is dropped rather than
    rendered as a blank numbered line, and the list is cut to `MAX_FACTS`. Both
    are the same answer to the same question: what is placed is what a reader
    of the prompt can see, and the count the receipt reports is of those.
    """
    if not facts:
        return ()
    kept: list[Fact] = []
    for fact in facts[:MAX_FACTS]:
        text = clean_fact_text(fact.text)
        if not text:
            continue
        kept.append(Fact(id=fact.id.strip(), text=text, values=tuple(fact.values)))
    return tuple(kept)


def validate_facts(facts: Sequence[Fact]) -> None:
    """Raise `FactError` on a facts list this service will not place.

    The ceilings and nothing else. What a fact means is the application's
    business; how much of a context window it may take is ours.
    """
    if len(facts) > MAX_FACTS:
        raise FactError(f"at most {MAX_FACTS} facts, got {len(facts)}")
    for fact in facts:
        if len(fact.id) > MAX_FACT_ID_CHARS:
            raise FactError(f"fact id is over {MAX_FACT_ID_CHARS} characters: {fact.id[:80]!r}")
        if len(fact.text) > MAX_FACT_TEXT_CHARS:
            raise FactError(f"fact {fact.id!r} text is over {MAX_FACT_TEXT_CHARS} characters")


def number_tokens(text: str) -> list[str]:
    """Every number-shaped token of a piece of text, in order, repeats kept.

    The written form rather than the value, because the report hands these
    back to a reader who has to find them in the answer: "0.667" and "66.7%"
    are one number and two different strings, and the one a member can search
    for is the one that was written.
    """
    found: list[str] = []
    for match in _NUMBER_TOKEN.finditer(text):
        token = match.group(0).rstrip(_TRAILING)
        if token and _PLAIN_NUMBER.fullmatch(token):
            found.append(token)
    return found


def parse_number(token: str) -> tuple[float, int] | None:
    """One token as a value and the number of decimal places it was written to.

    The places are what makes a rounded quotation verifiable: an answer that
    writes 66.7% against a row holding 0.6666666 is quoting the row, and the
    only honest way to say so is to round the row to the precision the answer
    used rather than to compare two floats that will never be equal.
    """
    body = token[:-1] if token.endswith("%") else token
    body = body.replace(",", "")
    try:
        value = float(body)
    except ValueError:  # pragma: no cover - the pattern already forbids this
        return None
    _, _, fraction = body.partition(".")
    return value, len(fraction)


def candidates(token: str) -> tuple[tuple[float, int], ...]:
    """The values one written number could be claiming, each with its precision.

    Two for a percentage and one for anything else. `60%` is 60 of something
    and it is also the rate 0.6, and the marts store rates as the second, so a
    check that knew only one of the two would report every correctly quoted
    win rate as unverified. The rate carries two decimal places more than the
    percentage was written to, because that is what dividing by a hundred
    does: `66.7%` is the rate 0.667 and not the rate 0.7.
    """
    parsed = parse_number(token)
    if parsed is None:
        return ()
    value, places = parsed
    if token.endswith("%"):
        return ((value, places), (value / 100, places + 2))
    return ((value, places),)


def _numbers_in(value: Any) -> list[float]:
    """Every number one evidence cell contributes, whatever type it arrived as."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if value is None:
        return []
    return [parsed[0] for token in number_tokens(str(value)) if (parsed := parse_number(token))]


def known_numbers(
    rows: Iterable[Mapping[str, Any]] = (),
    cards: Iterable[str] = (),
    facts: Sequence[Fact] = (),
    allowlist: Sequence[float] = ALLOWED_NUMBERS,
) -> frozenset[float]:
    """Every number an answer is allowed to state, from the four places it may come from.

    `rows` is every cell of every row any query returned, numbers and the
    numbers inside strings alike. `cards` is each card's printed fields run
    together, which is where a hit point total and an attack's damage live.
    `facts` is the `values` the application sent. The allowlist is the last,
    and the turn count joins it when the facts carry one.
    """
    known = {float(value) for value in allowlist}
    for row in rows:
        for cell in row.values():
            known.update(_numbers_in(cell))
    for card in cards:
        known.update(_numbers_in(card))
    for fact in facts:
        # The turn count is in here with everything else rather than in the
        # allowlist beside the four constants, because the application sends
        # it as a fact (`TURN_COUNT_FACT_ID`) and a number that is already a
        # fact value does not need a second way of being allowed.
        known.update(float(value) for value in fact.values)
    return frozenset(known)


def covered(token: str, known: Iterable[float]) -> bool:
    """Whether one written number is any of the numbers the run can account for.

    Two ways of being the same number, and the second is the one that makes
    a receipt readable: equal to within a float's worth of noise, or equal
    once the known value is rounded to the precision the answer wrote. An
    answer quoting 66.7% of a row holding 0.6666666 is quoting the row.
    """
    wanted = candidates(token)
    for value in known:
        for claim, places in wanted:
            if abs(value - claim) <= TOLERANCE or round(value, places) == claim:
                return True
    return False


def remembered_numbers(history: Iterable[str] = ()) -> frozenset[float]:
    """Every number the assistant's earlier turns in this conversation stated.

    A fourth place a number in an answer can have come from, and the one that
    is not evidence: an earlier answer is this agent's own words, written out
    of rows that were read in another request and are not in front of it now
    (rule 11 of the prompt). Only the assistant's turns are read, because a
    number a member typed is not a number the agent may repeat as a finding.
    """
    found: set[float] = set()
    for text in history:
        found.update(_numbers_in(text))
    return frozenset(found)


def check_numbers(
    answer: str,
    *,
    rows: Iterable[Mapping[str, Any]] = (),
    cards: Iterable[str] = (),
    facts: Sequence[Fact] = (),
    history: Iterable[str] = (),
    allowlist: Sequence[float] = ALLOWED_NUMBERS,
) -> NumberCheck:
    """Every number in the prose that nothing the run read can account for.

    Deterministic, offline and cheap: one scan of the answer, one set of
    floats, no model. The result is a report and never a refusal, so a false
    positive costs a line on a receipt rather than an answer.

    `history` is the assistant's earlier turns, and it is searched last and
    reported apart. A number found there and nowhere else is not an invention
    and is not a finding either: it is this agent quoting itself, which rule
    11 allows only when the evidence behind it is fetched again. So it comes
    back as `from_history` rather than as `unverified`, and the two counts
    answer two different questions about the same answer.

    Duplicates are reported once, in the order they were written, because
    "9, 9, 9" is one thing wrong and three lines about it is a receipt nobody
    reads.
    """
    known = known_numbers(rows=rows, cards=cards, facts=facts, allowlist=allowlist)
    remembered = remembered_numbers(history)
    unverified: list[str] = []
    from_history: list[str] = []
    for token in number_tokens(answer):
        if token in unverified or token in from_history or covered(token, known):
            continue
        if covered(token, remembered):
            from_history.append(token)
            continue
        unverified.append(token)
    return NumberCheck(unverified=tuple(unverified), known=known, from_history=tuple(from_history))


def cite_facts(answer: str, facts: Sequence[Fact]) -> list[FactEvidence]:
    """The facts that were placed, each with whether the answer used it.

    Two ways of using one, because an answer may do either and both are
    honest. It may cite the fact by its position in the numbered list, which
    rule 10 allows and which `_FACT_REFERENCE` reads; or it may simply state
    one of the numbers the fact carries, which is the ordinary case. A fact
    with no values at all ("you never attacked in this game") can only be
    cited the first way, which is correct: there is no number in it to find.
    """
    numbered = {int(match.group(1)) for match in _FACT_REFERENCE.finditer(answer)}
    written = {
        parsed[0] for token in number_tokens(answer) if (parsed := parse_number(token)) is not None
    }
    evidence: list[FactEvidence] = []
    for position, fact in enumerate(facts, start=1):
        stated = any(
            any(abs(value - float(claim)) <= TOLERANCE for value in written)
            for claim in fact.values
        )
        evidence.append(
            FactEvidence(id=fact.id, text=fact.text, cited=position in numbered or stated)
        )
    return evidence
