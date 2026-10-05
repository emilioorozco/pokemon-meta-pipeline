# What would make Ask competitively useful

A research spike for PLA-202, written 2026-10-04 against the committed schema,
the committed prompt and the committed golden set. Nothing here was measured by
running the agent.

**What this is.** An honest account of the distance between what Ask answers
today and what a competitive Pokemon Trading Card Game player asks while
choosing a deck, preparing a matchup, deciding a tech slot, reviewing a loss
and tracking their own leaks. It judges every question against
`pipeline/marts_schema.py`, `pipeline/prompts.py`, `evals/golden.yaml` and the
dbt models, and it says where the answer is missing because the data is missing
rather than because the model is small.

**What this is not.** It is not a measurement of the model. No agent was run,
no provider was called, no AWS call was made, nothing under `data/` was read
and no external site was fetched. Section 3 designs a model comparison and
states its price; it reports no result, because there is none.

**Two corpus figures are the ticket's, not this checkout's.** The ticket states
about 256 community games and 175 full decklists. `README.md` still says 128
games, 67 of them with shared decklists, and nothing in a checkout can settle
which is current. Every judgement below is about grain and about shape, so it
holds at either size; where a count matters the text says which figure it used.

Contents:

1. [The questions that matter](#1-the-questions-that-matter) (46 rows)
2. [Data gaps, ranked by questions unlocked](#2-data-gaps-ranked-by-questions-unlocked) (7 gaps)
3. [The reasoning gap, design only](#3-the-reasoning-gap-design-only) (10 new questions, 6 parked items)
4. [The coaching framing](#4-the-coaching-framing) (2 hand-written reviews)
5. [A measurement plan](#5-a-measurement-plan) (12 proposed cases)
6. [Proposed tickets](#6-proposed-tickets) (12 tickets, top 3 marked)

---

## What Ask can reach today

The ground truth for every verdict in section 1. The SQL tool will run against
eight relations and no others (`pipeline.prompts.ALLOWED_TABLES`), and the
prompt describes exactly those eight:

| table | grain | what it answers |
| --- | --- | --- |
| `mart_matchups` | ordered (archetype, opponent archetype) | a pairing's record and `win_rate`, with `games` and `min_games_met` |
| `mart_archetype_weekly` | (archetype, ISO week) | a deck's record and `share_of_week` by week, with `games` (seats) and `week_games` |
| `mart_archetype_pace` | archetype | ten tempo averages: first attack turn, share of turns without an attack, energy per turn, prizes by turns 4, 6, 8 and 10, first prize turn, first knockout turn, concession turn |
| `mart_cards_seen` | (archetype, card) | `seen_rate` (observation), `inclusion_rate` (decklist-backed seats only), `avg_copies_seen`, `max_copies_seen` |
| `mart_player_summary` | member | one row per member: record, the same record split by which side opened, favourite archetype, first and last seen |
| `dim_archetype`, `dim_card`, `dim_date` | the keys | names, aliases, card catalogue columns, ISO week arithmetic |

Plus `lookup_cards`, a retrieval tool over printed card text, which is a
reference and never game data.

Three facts about that list decide most of section 1. `fct_game_side` is not on
it, so nothing keyed by (game, seat) is reachable, which rules out every
question needing `went_first`, `mulligans`, `damage_dealt` or `cards_drawn` by
archetype. `mart_player_summary` is the only table keyed by a person, and it
holds one row per member with no archetype, week or opponent split. And no
table holds one game, which is why a post-loss review is answered out of the
`<facts>` list the application sends with the question and not out of the
warehouse at all.

`min_games` is a dbt variable defaulting to 5 (`dbt/dbt_project.yml`). A row
with 4 games is still returned; `min_games_met` reads false and rule 2 of the
prompt requires the agent to say so in words. Five games is not a sample. A
5 and 0 record has a 95 percent interval running from roughly 48 percent to
100 percent, and a 3 and 2 record covers nearly the whole range. Every "partly"
below that rests on a matchup cell inherits that.

---

## 1. The questions that matter

Forty-six questions, grouped by the moment a competitive player asks them.
They are representative of competitive preparation and of public coaching
framing, written from knowledge of how players prepare; they are not quotes
from any person or source. The verdict column is judged against the tables
above and against what `evals/golden.yaml` already proves the agent does.

`yes` means a competent run answers it today with the honesty rules on.
`partly` means an answer exists but is weaker than the question deserves, and
the last column says how. `no` means the warehouse cannot answer it and the
right behaviour is to say so.

### Choosing a deck for an event

| id | question | today | mart or fact | why it falls short |
| --- | --- | --- | --- | --- |
| Q01 | What is the strongest deck in the format right now? | partly | `mart_archetype_weekly` | "Right now" is one ISO week, which on this corpus is a few dozen games; most rows have `min_games_met` false, so the honest answer is a list of thin cells rather than a ranking |
| Q02 | What were the most played decks last week, and what share of the field was each? | yes | `mart_archetype_weekly.share_of_week`, `week_games` | Share sums to about two across a week because every game has two decks; the agent has to say so or the number reads double |
| Q03 | Has this deck been rising or falling over the last four weeks? | partly | `mart_archetype_weekly` by `week_start` | Four weekly cells of a handful of games each is noise with a shape; nothing in the mart distinguishes a trend from a run of coin flips |
| Q04 | Which deck has the best record against the three most played decks? | partly | `mart_matchups` joined to weekly share | One SELECT can do it, but it averages three thin cells and the result carries no interval |
| Q05 | If the field is mostly A and B, what is my expected win rate with C? | no | would need `mart_matchups` weighted by a field prior | No table holds an expected field; the agent would have to be handed the weights in the question, and nothing offers that |
| Q06 | Which deck is the safest choice if I have not practised much? | no | none | Nothing measures difficulty, variance or how a deck's record moves with pilot experience |
| Q07 | What did the top finishers play at the last three events? | no | none | No event grain anywhere, and no external tournament data; see gap G5 |
| Q08 | Is this card legal in the current format? | partly | `dim_card.catalog_reg`, `lookup_cards` | The regulation mark is per card and the catalogue may not be fetched (`in_catalog`); `dim_format` is a placeholder and is not on the allowlist, so nothing says what the current format is |
| Q09 | How much does going first matter for this deck? | no | `fct_game_side.went_first`, not allowlisted | The per-side split exists only on `mart_player_summary`, which is per member; no archetype split exists; see gap G4 |
| Q10 | Which deck gives me personally the best record? | partly | `mart_player_summary.favourite_archetype_*` | The mart holds one win rate per member and the deck they played most, not a record per deck; see gap G1 |

### Preparing a matchup

| id | question | today | mart or fact | why it falls short |
| --- | --- | --- | --- | --- |
| Q11 | How does A do against B, and over how many games? | yes | `mart_matchups` | Proven by `matchup_win_rate` in the golden set |
| Q12 | Is that matchup number trustworthy? | yes | `mart_matchups.min_games_met` | Proven by `matchup_thin_sample` |
| Q13 | Which decks beat A most often? | yes | `mart_matchups` filtered on `opponent_archetype_key` | Rule 6 forces every tied row to be named, which on this corpus is often three rows at 100 percent over one game each |
| Q14 | Which of A's bad matchups are common enough to care about? | partly | `mart_matchups` plus `mart_archetype_weekly` | The two grains are seats and weeks; the join is writable but the agent has to reconcile `games` against `week_games`, and both sides are thin |
| Q15 | What cards should I expect to see from B? | partly | `mart_cards_seen` | `seen_rate` is an observation and a lower bound (rule 3); `inclusion_rate` is null for any archetype with no decklist-backed seats |
| Q16 | How many copies of a card do B's lists run? | no | `avg_copies_seen`, `max_copies_seen` | Those are copies observed in play, capped by the producer, not copies in a list; silver writes no row per decklist card; see gap G3 |
| Q17 | What tech cards have started showing up in B? | no | `mart_cards_seen` has no time grain | The mart is (archetype, card) over the whole corpus, so "lately" has nowhere to come from |
| Q18 | Does B win the prize race early or late? | partly | `mart_archetype_pace.prizes_by_turn_4/6/8/10` | Averaged over all of B's seats against every opponent, so it cannot say what B does against me |
| Q19 | Who attacks first in A against B, and does that decide it? | partly | `mart_archetype_pace.first_attack_turn` | Pace is per archetype, never per matchup, and nothing relates a pace number to a result; see gap G4 |

### In-event tech decisions

| id | question | today | mart or fact | why it falls short |
| --- | --- | --- | --- | --- |
| Q20 | I am 2 and 1; what am I likely to face next round? | no | none | No standings, no event, no pairings |
| Q21 | Is a counter card for B worth the slot this weekend? | no | would need card-level records | Needs "seats that ran X did Y", which needs decklist rows; see gap G3 |
| Q22 | Is B played enough to tech for? | yes | `mart_archetype_weekly.share_of_week` | Honest, with the doubling caveat and the games count |
| Q23 | What is the most played card I am not playing? | no | none | Nothing holds the member's own list |
| Q24 | If I switch decks for the last rounds, what do my matchups look like? | partly | `mart_matchups` for the new deck | The community's rows, not the member's; thin cells |
| Q25 | Do I win more going second with this particular deck? | no | none | Needs member by archetype by side, which is three grains past `mart_player_summary`; see gap G1 |
| Q26 | What exactly does this card do? | yes | `lookup_cards`, `dim_card` | Proven by `card_text_lookup`; the `card_rules` playbook puts printed text first |

### Post-loss review

| id | question | today | mart or fact | why it falls short |
| --- | --- | --- | --- | --- |
| Q27 | Why did I lose that game? | partly | the `<facts>` list, `my_mistake` playbook | The facts describe; they do not explain. No board state, no hand, no prize values per knockout |
| Q28 | Which turns did I not attack on? | yes | the turns-without-attack fact | Proven by `mistake_game_01`, with `max_unverified: 0` |
| Q29 | Was I behind when I conceded? | partly | prizes taken, prizes by turn, concession turn | Prize counts are not the position: a level prize count with a board behind is a loss, and the log holds no board |
| Q30 | Was that loss normal for this matchup, or did I misplay? | partly | `mart_matchups`, `mart_archetype_pace` | The matchup rate and the pace averages answer "normal for the deck"; neither is conditioned on the pairing, and the member's own deck is often not named in the context |
| Q31 | Was their turn-3 knockout lucky or standard? | no | `mart_archetype_pace.first_knockout_turn` is an average | An average cannot say whether one game sat in the tail; no distribution and no per-matchup row |
| Q32 | Which single decision cost me the game? | no | none | No counterfactual, no legal-move model, no board. The right answer is to say so, which is the subject of gap G6 and ticket T2 |
| Q33 | How did my pace compare with other players of my deck? | partly | the facts plus `mart_archetype_pace` | Only possible when the member's own archetype is named in the context, which it often is not; and the comparison is to a mean with no spread |
| Q34 | Did I lose to the matchup or to how I played it? | partly | the same two sources | Nothing separates them; the honest answer names both and attributes neither |
| Q35 | Show me my three games most like this one. | no | none | No game-level table is on the allowlist, and no similarity exists anywhere |

### Tracking personal leaks

| id | question | today | mart or fact | why it falls short |
| --- | --- | --- | --- | --- |
| Q36 | What is my record? | yes | `mart_player_summary` filtered on the token | Shipped with PLA-208; the application states the token in the page context |
| Q37 | Which deck have I lost to most? | no | none at that grain | An offered chip today. `mart_player_summary` has no opponent split, and `mart_matchups` is the community's record; the playbook makes the agent say so, which is accurate and unsatisfying |
| Q38 | How does going first change my win rate? | yes | `games_first`, `wins_first`, `games_second`, `wins_second` | Shipped with PLA-208; both halves with the games behind each |
| Q39 | How has my win rate moved week by week? | no | none at that grain | An offered chip today; no member by week row exists |
| Q40 | How often do I take a turn with no attack, against other pilots of my deck? | no | none at that grain | The pace mart is per archetype; no per-member pace row exists, although `int_game_side_pace` already computes the numbers per seat |
| Q41 | Am I slow to my first prize compared with the field? | partly | one game's fact against `mart_archetype_pace` | One game against an average is an anecdote against a mean; no per-member average across their games |
| Q42 | Do I lose more when I am behind on prizes at turn 6? | no | none | Needs the per-game prize map joined to the result across a member's games |
| Q43 | Which of my mistakes repeats? | no | none | Facts live in the request and are never stored or aggregated; nothing can count a pattern across games |
| Q44 | Am I getting better? | no | none | No member time series of any kind |
| Q45 | How many games are behind all of this? | yes | any mart's `games` or `week_games` | Rule 1 makes it compulsory anyway |
| Q46 | Who is the strongest player in the league? | no, by design | `dim_player` is not readable | Rule 5: handles become one-way tokens before anything is written, so there is no answer to give. A correct refusal, and the one place "no" is the product working |

**Tally.** 10 `yes`, 16 `partly`, 20 `no`, one of the twenty refused by
design. The
shape of the result is the finding: Ask is good at the questions the marts were
built for, which are the community's aggregates, and weak at the two moments a
competitive player actually opens a tool, which are "what should I bring" and
"what did I do wrong". Thirteen of the nineteen "no" rows that are not refusals
by design are missing a grain,
not missing intelligence.

---

## 2. Data gaps, ranked by questions unlocked

Size is S (a day or two), M (about a week), L (longer, or blocked on something
outside the repository).

| rank | gap | questions unlocked | count | size | unlocked per unit of work |
| --- | --- | --- | --- | --- | --- |
| 1 | G1 per-member grain | Q10, Q25, Q37, Q39, Q40, Q41, Q42, Q44, and the quality of Q36 | 8 | M | high |
| 2 | G7 widen the facts the request carries | Q27, Q29, Q32 (partly), Q34, Q42 (half) | 4 | S | high |
| 3 | G4 turn-level marts beyond pace | Q09, Q18, Q19, Q30, Q31, Q33 | 6 | M | high |
| 4 | G2 sample size and intervals | none new; repairs 16 `partly` rows | 0 | S | high, as honesty rather than coverage |
| 5 | G3 decklist-level data | Q15, Q16, Q17, Q21, Q23, Q05 (half) | 5 | M to L | medium |
| 6 | G6 reconstructed-tier facts | Q29, Q31, Q32, Q34 | 4 | M, mostly in the application | medium |
| 7 | G5 external meta sources | Q01, Q04, Q05, Q07, Q17, Q20 (partly), Q21 | 6 | L, licence-gated | low until the licence question is answered |

### G1. Per-member grain

**What has to be built.** Four marts over `fct_game_side` and
`int_game_side_pace`, no new ingestion and no new silver column:
`mart_player_archetype` (member by archetype by opening side),
`mart_player_matchup` (member by opponent archetype), `mart_player_weekly`
(member by ISO week), and `mart_player_pace` (the ten pace numbers averaged
over a member's seats, so a member's own number sits beside the archetype
average that `mart_archetype_pace` already holds). Each is a group-by the fact
table already supports. The allowlist and the generated schema listing grow by
four tables.

**Size.** M. The SQL is routine; the cost is in the tests, the schema
descriptions, the prompt budget and the playbook edits.

**Risks.** Three, and the first is the real one. A member with 40 games split
by archetype and by opponent produces cells of one and two games, so every new
mart is a thin-cell factory and rules 1 and 2 become most of the answer; this
is the gap that most needs G2 shipped beside it. Second, the prompt has a
ceiling: `MAX_PROMPT_CHARS` is 21,600 against about 19,700 rendered, so four
new tables do not fit without raising it, and the cached prefix grows on every
request. Third, a per-member per-week per-archetype row is a finer grain than
anything `docs/data-handling.md` has reasoned about; the token is still
irreversible and the rows are still members only, but the maintainer should
confirm the grain before it ships.

### G2. Sample size, and what `min_games` means

**The state.** `min_games` is 5. `mart_matchups`, `mart_archetype_weekly` and
`mart_archetype_pace` each carry `min_games_met`, and the prompt makes the
agent repeat it. That is honest and it is not useful: a player told "60 percent
over 5 games, and the sample is thin" has been told nothing they can act on.

**What has to be built.** A Wilson score interval beside every rate:
`win_rate_low` and `win_rate_high` on `mart_matchups` and
`mart_archetype_weekly`, computed in SQL from `wins` and `losses`, plus one
prompt rule saying the interval goes in the same sentence as the rate. Nothing
upstream changes.

**Size.** S. Two dbt columns, two schema descriptions, one rule, two golden
cases.

**Risks.** The rule is a twelfth rule in a prompt whose ceiling is already
tight. And an interval is a different kind of honesty: "between 29 percent and
85 percent" is a truthful sentence that some members will read as the tool
being broken, so the phrasing matters more than the arithmetic. The upside is
that it converts sixteen weak answers into answers a player can weigh, which
is why it sits above three larger gaps.

### G3. Decklist-level data

**The state.** Decklists arrive in bronze as a structure per seat, and silver
keeps only three derived columns per seat (`decklist_source`,
`decklist_complete`, `decklist_card_count`) plus a boolean `in_decklist` on each
observed card. There is no row per decklist card anywhere. So `avg_copies_seen`
is copies observed in play, and `inclusion_rate` is still bounded by
observation. The ticket says 175 full lists have landed; nothing in the
warehouse can use them as lists.

**What has to be built.** A silver table at (game, seat, card) with a count,
exploded from the bronze decklist structure; a mart `mart_decklist_cards`
(archetype by card: how many lists hold it, the distribution of counts, the
modal count); and, for the question players actually ask, a
`mart_card_contrast` (archetype by card: the record of seats that ran it
against the record of seats that did not). The card key bridging already exists
in silver and depends on the catalogue being fetched.

**Size.** M for the silver table and the first mart, L with the contrast mart.

**Risks.** 175 lists spread over a dozen archetypes is a handful of lists per
deck, so a modal count is reportable and a contrast is not: a card-level win
rate over six lists is the most confidently wrong number this project could
produce, and it would read as the most authoritative thing on the page. If the
contrast mart is built, it needs a threshold well above `min_games` and a rule
of its own. Second risk: `in_decklist` reads false everywhere when the
catalogue was not fetched before the silver run, so the pipeline's optional
dependency becomes load bearing.

### G4. Turn-level patterns beyond the pace mart

**The state.** Silver already writes a `turns` table, one row per turn segment
with action counters by kind, and `fct_game_side` already carries `went_first`
and `mulligans`. `mart_archetype_pace` is one group-by over that material.
Everything in this gap is another group-by over data that is already landed.

**What has to be built.** `mart_archetype_first_turn` (per archetype, the
record split by which side opened, which is Q09 on its own and is the smallest
useful thing on this list); `mart_matchup_pace` (the ten pace numbers per
ordered pairing, which is Q18 and Q19); `mart_prize_map` (archetype by turn,
average prizes taken by each side, which gives a shape rather than four
snapshots); and a mulligan split on the archetype record.

**Size.** M for all four, S for the going-first split alone.

**Risks.** Cells. A pairing with 5 games has two or three on each side of the
coin, so `mart_matchup_pace` is mostly `min_games_met` false, and a mart that
is always flagged thin teaches members to ignore the flag. The prompt budget
again: four more tables is another schema block. And the pace definitions are
written twice already, once in the application and once in SQL, held together
by a test over the ten fixture games; every new turn-level mart is a third
place a definition can drift.

### G5. External meta sources

**The licence comes first, and it is not a task, it is a gate.** Before any
ingestion is designed, somebody has to read the source's terms of use and
answer four questions in writing: may results be ingested and stored, may
decklists be ingested and stored, what attribution is required and where it
must appear, and is a commercial or a members-only product in scope. Nothing in
this spike checked any of that; no external site was fetched. Until those four
answers exist, this gap has a size of "unknown" and a plan of "do not start".

**What would have to be built, if permitted.** A separate ingestion stage
writing to a separate namespace, so an external row and a community row can
never be confused by a join; marts named so that the source is in the name; a
twelfth and thirteenth prompt rule saying that external numbers are labelled as
external in every sentence that uses one, and that an external rate and a
community rate are never averaged, added or compared without naming both
sources and both sample sizes; attribution rendered by the application on any
answer that used an external row; and an adversarial golden case that tries to
get the two averaged together.

**Size.** L.

**Risks.** The licence risk is the obvious one. The subtler one is to the
product's only real claim. This project's distinguishing honesty is that it is
precise about a small corpus; an answer that silently mixes a 5-game community
cell with a 400-game external cell is not a better answer, it is the end of the
claim. The separation has to be in the schema and in the rules, not in the
model's good judgement.

### G6. Reconstructed-tier facts

**The state.** The application's fact schema declares two tiers and requires a
caveat string on the reconstructed one. Today every fact it computes is in the
exact tier, so the reconstructed tier has no members. Facts such as damage on
the board or energy on the active Pokemon need an assumption (what a damage
counter is worth, hit points looked up by printed name), which is exactly what
the tier was declared for.

**What has to be built.** Mostly in the application: the facts themselves, with
their caveats, and a catalogue version bump with a backfill. In this repository:
one glossary line per new fact in `FACTS_GLOSSARY` (the glossary is held to the
fixture fact ids by a test, so a new fact without a line is a red test), a
prompt rule that a reconstructed fact is cited with its caveat, regenerated
`evals/fixtures/facts/`, and at least two new `mistake` cases that fail when
the caveat is dropped.

**Size.** M, with most of it outside this repository.

**Risks.** One that matters. Rule 10's numeric check passes any number that
appears in a fact's `values`, so a reconstructed number that is wrong is a
wrong number with a clean receipt. The check cannot see it, by construction
(`pipeline/facts.py` says so). That makes the caveat the only guard, and
caveats are the first thing a model drops under a word limit, which is why
the golden cases have to grade the caveat and not only the number.

### G7. Widen the facts the request carries

**The state.** The service accepts up to 60 facts per request. The fixtures and
the golden `mistake` cases carry thirteen. The application computes more per
game than it sends, so some of what a post-loss review needs is already being
computed and is simply not in the request.

**What has to be built.** Nothing in the data pipeline. A request-shape change
on the application's side, one glossary line per fact added here, regenerated
fixtures, and new `mistake` cases.

**Size.** S, and it is the cheapest real improvement in this document.

**Risks.** More numbers in front of the model is more surface for an invented
number, which is what rule 10's check exists to catch, so the check's
`unverified_numbers` series is the thing to watch after it ships. The human
turn grows, which costs uncached input tokens on every call of that question
(the cached prefix is unaffected), and the glossary grows against
`MAX_PROMPT_CHARS`.

---

## 3. The reasoning gap, design only

**This has not been run.** No provider call was made for this spike. What
follows is the design, the commands and the price.

### What to run

**The cases.** All 52 of `evals/golden.yaml`, unchanged, plus 10 new
competitive questions. The 52 are what makes the numbers comparable with every
run already in MLflow, and the 14 adversarial cases plus the two injection
cases are the control: a stronger model that becomes more helpful about a
prompt injection is a worse model here, and the comparison has to be able to
see that.

**One constraint on the 10 new cases, which shapes how they are written.**
`--model` and `--remote` are refused together by `pipeline.eval`, deliberately,
so a model comparison runs locally against the fixture warehouse. The ten
questions therefore have to be answerable from the ten fixture games, which
means they are written as `warehouse: any` with shape-based `require` entries
(a rate with a sample size beside it, a named archetype with a count) rather
than as facts of the production corpus.

**The ten new competitive questions.**

| id | job | question |
| --- | --- | --- |
| N01 | meta | I can bring one of two decks this weekend. Based on the last two weeks, which has the better expected record, and how confident should I be? |
| N02 | meta | Which decks beat Dragapult control most often, and are any of them actually being played? |
| N03 | meta | How quickly does Dragapult control take its first prize compared with the rest of the field, and what does that mean for playing against it? |
| N04 | meta | How often does a card turn up in the lists of the deck I expect to face, and is that often enough to play around? |
| N05 | my_record | Do I win more going first, and is the difference big enough to care about? |
| N06 | my_record | Which deck have I lost to most, and if you cannot tell me that, what is the closest thing you can tell me? |
| N07 | my_mistake | I lost that one with prizes level. What decided it, and what would you do differently? |
| N08 | my_mistake | My first attack was early and my first prize was late. Is that a me problem or a matchup problem? |
| N09 | my_game | Was the pace of that game normal for this matchup? |
| N10 | meta | What are people playing at tournaments right now? |

They are chosen so that lookup is not the hard part. N01 asks for a weighing,
N03 for an inference from a tempo number to a plan, N06 and N10 for a refusal
that still leaves the member better off, N08 for an attribution the data cannot
make. If a bigger model is worth money anywhere, it is worth it here.

### The rubric

Four axes, 0 to 2 each, 8 maximum per question. Scored by hand, by a scorer
reading the answers with the model name stripped and the order shuffled.

| axis | 2 | 1 | 0 |
| --- | --- | --- | --- |
| Correct | every number traces to a row, a card or a fact, and is the right number for the question's grain | one number is untraceable or comes from the wrong grain | a number is invented, or the claim is wrong |
| Actionable | a player could change a deck choice, a tech slot or a line because of it | true, and not usable | restates the question or the data model |
| Cites the right evidence | reads the mart or the fact whose grain matches the question, and says which | reads something adjacent and does not say so | answers with no evidence, or with the wrong table |
| Names the uncertainty | the sample size and the thin-cell flag are in the same sentence as the claim | mentioned somewhere in the answer | absent, or a rate given with no denominator |

Two reasons for this rubric rather than the harness's pass or fail. The
harness's `require` entries are deliberately about facts and never about
wording (`evals/golden.yaml` says so), so they cannot see the difference
between a correct sentence and a useful one. And "names the uncertainty" is
precisely the axis a bigger model might regress on, because a more fluent
answer is a more confident answer.

Scores are recorded in a CSV beside the MLflow run id, with the question id,
the model, the four axis scores and one line of justification. Not in MLflow:
these are hand scores, and a hand score logged as a metric reads later as
something the harness computed.

### Cost and latency, from the existing harness

**Cost is derivable and is not reported today.** `Report.as_dict` carries
`usage_totals`, summed over every question, under the provider's own names
(`input_tokens`, `output_tokens`, `cache_read_input_tokens`,
`cache_creation_input_tokens`), and the MLflow run logs `cache_read_tokens` and
`cache_creation_tokens`. `gate_cost_usd` is the only dollar figure the harness
prints and it is the optional SQL gate's cost, not the answering model's. So
the comparison prices the run by hand from the `--json` report at list prices.
That is a four-line script over the report and needs no network.

**Latency is not captured at all, and that is a gap in the harness.** The
deployed service puts `latency_ms` in the `/ask` body, but `--model` cannot be
combined with `--remote`, and the local path never times anything:
`pipeline.eval.Result` has no duration field. Three options, in order of
preference: add an elapsed field to `Result` and print it in the report, which
is a small change to `run_question` and the one worth making; or time each
model's whole run with the shell and divide by the case count, which is crude
but comparable across models since the case set is identical; or accept that
this comparison reports cost and not latency and open a ticket. The first is
recommended, because a routed model choice is a latency decision as much as a
cost one, and "about 5 to 20 seconds" is the only latency figure the product
currently states to a member.

**One thing to decide before running, because it moves both numbers.** The
agent builds its provider client with a 60 second timeout and no thinking
configuration (`pipeline.agent.chat_model`). Haiku 4.5 does no thinking unless
asked. Sonnet 5 and Opus 5 run adaptive thinking, and on Opus 5 it is on by
default, so both will emit thinking tokens, billed as output, on every call of
every question, and both will be slower. The comparison should either set a low
effort for the two larger models and say so, or raise the client timeout, or
both. Running them at defaults against a 60 second timeout risks measuring the
timeout.

**Prefix caching is comparable across all three.** The prefix is about 5,200 to
5,600 tokens (measured 2026-10-04: 4,299 for the system blocks as they then
stood, about 645 for the `query_marts` tool schema, and the facts glossary
since). The minimum cacheable prefix is 4,096 tokens on Haiku 4.5, 1,024 on
Sonnet 5 and 512 on Opus 5, so the prefix caches on all three and the cheapest
model is the one with the least headroom over its minimum. Writes bill at 1.25
times input and reads at 0.1 times, on the five-minute entry the service uses.

### The commands

Build the fixture warehouse once, exactly as `docs/evals.md` describes, into a
scratch directory and never into `data/`:

```bash
export PIPELINE_DATA_DIR=/tmp/pla202-eval
HANDLE_HMAC_KEY=$(openssl rand -hex 32) uv run python -m pipeline.backfill \
  --source-dir tests/fixtures \
  --bronze-dir "$PIPELINE_DATA_DIR/lake/bronze" \
  --quarantine-dir "$PIPELINE_DATA_DIR/lake/quarantine"
uv run python -m pipeline.silver \
  --bronze-dir "$PIPELINE_DATA_DIR/lake/bronze" \
  --silver-dir "$PIPELINE_DATA_DIR/lake/silver" \
  --catalog tests/catalog.json
uv run python -m pipeline.gold --data-dir "$PIPELINE_DATA_DIR"
uv run python -m pipeline.card_index build --source tests/card_text.jsonl \
  --out "$PIPELINE_DATA_DIR/card_index" --embedder hashing
```

Then the same set against each model, with everything else held identical. The
harness takes the model by name and hands it to `pipeline.agent`:

```bash
mkdir -p /tmp/pla202-eval/runs
for MODEL in claude-haiku-4-5-20251001 claude-sonnet-5 claude-opus-5; do
  op run --env-file=.env.op -- uv run python -m pipeline.eval \
    --model "$MODEL" \
    --golden evals/golden.yaml \
    --warehouse "$PIPELINE_DATA_DIR/warehouse/meta.duckdb" \
    --card-index "$PIPELINE_DATA_DIR/card_index" \
    --experiment agent-model-comparison \
    --json > "/tmp/pla202-eval/runs/golden-$MODEL.json"
done
```

And the competitive subset, once it exists as a file of its own:

```bash
for MODEL in claude-haiku-4-5-20251001 claude-sonnet-5 claude-opus-5; do
  op run --env-file=.env.op -- uv run python -m pipeline.eval \
    --model "$MODEL" \
    --golden evals/competitive.yaml \
    --warehouse "$PIPELINE_DATA_DIR/warehouse/meta.duckdb" \
    --card-index "$PIPELINE_DATA_DIR/card_index" \
    --experiment agent-model-comparison \
    --json > "/tmp/pla202-eval/runs/competitive-$MODEL.json"
done
```

Pricing a finished run from its report, with no network:

```bash
uv run python - <<'PY'
import json, pathlib
PRICE = {  # dollars per million tokens, list prices as of 2026-10-04
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
}
for path in sorted(pathlib.Path("/tmp/pla202-eval/runs").glob("*.json")):
    report = json.loads(path.read_text())
    usage, model = report["usage_totals"], report["model"]
    rate_in, rate_out = PRICE[model]
    cost = (
        usage["input_tokens"] * rate_in
        + usage["cache_creation_input_tokens"] * rate_in * 1.25
        + usage["cache_read_input_tokens"] * rate_in * 0.1
        + usage["output_tokens"] * rate_out
    ) / 1e6
    print(f"{path.name:44s} {model:30s} ${cost:7.4f}  {report['passed']}/{report['total']}")
PY
```

The broken-prompt control is worth one more run, on the cheapest model only, to
confirm the rules are still load bearing after any prompt edit this spike
causes: `--prompt-override evals/broken_prompt.txt`.

### What it will cost

Estimated, not measured, from the prompt size this repository records and the
question shape the ticket states: a cached prefix of about 5,200 tokens, 2 to 4
model calls per question, 12,000 to 19,000 input tokens per call in total, and
300 to 500 output tokens per call. The two larger models are costed with 600
extra output tokens per call for adaptive thinking, which is a guess and the
single largest source of error in this table.

Per question, at list prices:

| model | lean (2 calls, 7,000 uncached input and 350 output per call) | heavy (4 calls, 13,800 uncached input and 500 output per call) |
| --- | ---: | ---: |
| Haiku 4.5 | $0.025 | $0.073 |
| Sonnet 5 | $0.061 | $0.171 |
| Opus 5 | $0.153 | $0.426 |

For the whole comparison, 62 cases on three models, at the midpoint of those
two scenarios:

| repetitions | model runs | estimated total |
| --- | ---: | ---: |
| 1 | 186 | about $28 |
| 2 | 372 | about $56 |
| 3 | 558 | about $84 |

Two repetitions is the recommendation: one run cannot separate a model
difference from a sampling difference on a set this small, and three doubles
the bill for a third data point. At two repetitions the split is roughly $6 on
Haiku 4.5, $14 on Sonnet 5 and $36 on Opus 5. The whole exercise is cheaper
than an hour of anybody's time, which is the actual argument for running it
rather than reasoning about it.

**The decision the numbers are for.** Whether to route by job: the cheap model
for `meta`, `card_rules` and the lookup half of `my_record`, a stronger one for
`my_mistake` and for matchup preparation. The rule proposed in advance, so the
result cannot be read to taste: route by job only if the stronger model gains at
least 1.5 rubric points of 8 on the `my_mistake` and competitive subsets while
losing nothing on the adversarial set. Below that, the money belongs in the
data gaps.

### The items already parked on this ticket

| parked item | where it belongs |
| --- | --- |
| Stored server-side conversations | Not in this plan, and not yet a ticket. It unlocks none of the 46 questions, and it costs the sentence in `docs/agent-safety.md` saying nothing a member typed is stored, plus a consent line and a retention policy. Decide it from the one-week trial in section 5: if the two players ask across sittings and say the loss of context hurts, write the privacy wording first and the checkpointer second. |
| Rolling conversation summaries | A cost optimisation at a window nobody is known to have reached. The history cap is 6 turns and 6,000 characters, and the second cache breakpoint already makes a follow-up cheap. Belongs in the measurement plan as instrumentation: count the threads that hit the cap. Build only when the count is not zero. A summary is also not evidence, which rule 11 already says about an earlier answer, so the summary would need the same treatment. |
| A "me" resolution | Half shipped as PLA-208: the application states the member's player token in the page context and the agent filters `mart_player_summary` on it. What is left is not identity, it is grain, and it is gap G1 and ticket T1. |
| Making every offered question answerable | The `--offered` run already exists and is already in the `prod` job as a notice, with 7 of 70 failing on the last recorded run. It belongs as ticket T8: make the failures answerable, then turn the notice into a gate, with the rule that a chip ships only after it passes on prod. Most of the failures are the per-member and per-week chips that G1 unlocks, so T8 follows T1. |
| Better "I cannot answer that" answers | The prompt and the playbooks, plus a new golden kind. This is ticket T2 and it is in the top three, because twenty of the forty-six questions end in a refusal and the refusal is currently accurate and useless. It is also the clearest thing the model comparison will not fix: a bigger model given no instruction to offer the nearest answerable question will not reliably offer one. |
| An `allowed_low` chart with an unmatched number | Not a reasoning item. It is two decisions. First, whether a low-confidence gate verdict combined with `unverified_numbers` greater than zero should be rendered at all, which is a product rule and belongs with the gate's handling, not with the model. Second, whether a chart, which is a rendering of returned rows, may draw a number the prose does not trace to those rows; it should not, and the cheapest fix is to draw charts only from `evidence.queries[].rows`. It earns one rubric line ("names the uncertainty") and one eval column, and otherwise it is ticket T12. Note that the chart itself is not visible from this repository; see the open questions at the end. |

---

## 4. The coaching framing

### What a post-loss review should contain

Four parts, in this order, and nothing else:

1. **The three facts that mattered**, cited, chosen because they decided the
   game and not because they are the largest numbers. A fact that decided the
   game is one where the member's number and the opponent's number diverge, or
   where a clock started late.
2. **One line they could have taken**, written as a choice at a named turn and
   not as a verdict. "On turn 4 you attacked without attaching" is a fact; "on
   turn 4 the choice was between the attack and the attachment" is a line.
3. **What the matchup usually looks like**, with its games count, so the member
   can tell an unlucky game from a bad one. When the warehouse has no row, that
   sentence says so and is still worth writing, because "nobody has played this
   enough to say" is information.
4. **One thing to watch next game**, a single observable the member can check
   while playing. This is the part that turns a review into coaching, and it is
   the part nothing in the current playbook asks for.

And four things it must not contain: what the opponent held or drew, a verdict
on the member, a number with no row or fact behind it, and more than about 180
words.

### Fixture game A: a six-turn loss

Stock fixture, the `mistake_game_02` facts. Member went second and lost on turn
6, prizes 1 to 4, knockouts 1 to 3; first attack on turn 2 against the
opponent's turn 3; attacked on every turn taken and attached energy on every
turn taken; first prize on turn 4; first knockout on turn 4; biggest attack 200
damage on turn 4; energy 2 per turn; the game ended on a concession on turn 6.

**Hand-written review.**

> This was not a speed problem. You attacked first, on turn 2 against their turn
> 3, you attacked on every turn you took and you attached energy on every turn
> you took, at 2 per turn. The three facts that decided it are the trade: you
> took 1 prize and they took 4, you were credited with 1 knockout and they were
> credited with 3, and your biggest attack of the game, 200 damage on turn 4,
> bought one of those prizes.
>
> So the line to revisit is not when you attacked, it is what the turn-4 attack
> was for. A 200-damage attack that returns one prize against a deck taking
> three knockouts in six turns is the trade that lost the game. The log records
> the damage and the turn and not the target, so this is the one you have to
> supply: was that attack the biggest prize available, or only the biggest
> number.
>
> The warehouse holds no games for this pairing, so there is nothing to tell you
> whether six turns is fast for it.
>
> Next game, count the prizes at the end of each of your turns against theirs.
> This one finished 1 to 4.

**What `my_mistake` could say today, with the same inputs.** Most of it. The
playbook asks for exactly the first three parts, in that order, under 180 words,
with facts cited by number and an honest "the warehouse holds no row" for the
matchup, and rule 10's check would pass every number above because each is a
fact value. What it would not produce: the fourth part, because no playbook
asks for it; the sentence about the target, because nothing tells the model that
the absence of a target in the fact catalogue is itself worth saying; and the
framing "this was not a speed problem", because that is an inference from three
facts agreeing, and the playbook asks for the facts rather than for what they
agree on.

**What is missing from the data.** The prize value of each knockout, which is
what makes this a prize-trade loss rather than a damage loss; the target of an
attack; the board. With those, the second part stops being a question put back
to the member and becomes a line.

### Fixture game B: a seventeen-turn loss with prizes level

Stock fixture, the `mistake_game_04` facts, which is also the game behind the
`job_my_mistake_review` golden case. Member went second and lost on turn 17,
prizes 4 to 4, knockouts 2 to 4; first attack on turn 4 against the opponent's
turn 7; one turn with no attack (turn 2) and one turn with no energy attached
(turn 4); first prize and first knockout both on turn 12; biggest attack 200 on
turn 10; 1 energy per turn; a concession on turn 17. The pace oracle for the
same game records the opponent's first prize on turn 7, the turn they first
attacked.

**Hand-written review.**

> You were three turns ahead of them to the first attack, turn 4 against turn 7,
> and eight turns behind them to the first prize: theirs came on turn 7, yours
> on turn 12. That gap is the game. Prizes finished level at 4 each, and
> knockouts went 2 to you and 4 to them, which says their knockouts were worth
> more than yours were.
>
> The turn to look at is turn 4. It is the turn you first attacked and the only
> turn in the game you attached no energy, at an average of 1 energy per turn
> you could not afford to skip one. The choice there was between starting the
> attack clock and keeping the energy clock going, and the rest of the game says
> the energy mattered more: your biggest attack, 200 on turn 10, still did not
> start the prize count.
>
> The warehouse holds no row for this pairing, so there is no community number
> for how it usually goes.
>
> Next game, watch the gap between your first attack and your first prize. Eight
> turns is the number to beat.

**What `my_mistake` could say today, with the same inputs.** The first three
parts again, and this one is harder: the facts that decided it are turn 4 and
turn 12, and the golden case for this exact game requires both numbers, a line
written as a choice, and a sentence placing it against the matchup. So the
current playbook does reach the shape. What it would not produce: the turn-7
comparison, because the opponent's first prize turn is in the pace oracle and
not in the thirteen facts the request carries; the inference that knockouts
worth more prizes explains a level prize count with an uneven knockout count;
and the fourth part again.

**What is missing from the data, and one thing that is missing from the
request.** Missing from the data: prize value per knockout, and a per-matchup
pace row, so "eight turns from first attack to first prize" can be called late
or ordinary rather than just stated. Missing from the request: the member's own
archetype. The page context for both of these games says "Your deck against
<opponent archetype>", with the member's own deck unnamed, so the agent cannot
look up its pace row even though `mart_archetype_pace` would answer it. That is
a one-field change on the application's side and it is ticket T3.

---

## 5. A measurement plan

### A competitive golden subset

Twelve cases, proposed here and not yet in `evals/golden.yaml`. They are
written in the golden format with one addition: an `expected_points` list,
which is the expert's statement of what a good answer contains. The harness
ignores unknown fields today, so `expected_points` is documentation until
somebody teaches the scorer to print it beside a failure; the point of writing
it now is that the rubric in section 3 needs something to score against that
was written before anybody saw an answer.

All twelve are `warehouse: any`, so they can be run against the deployment as
well as the fixtures. None of them asserts a production number.

```yaml
# Proposed: evals/competitive.yaml, or kind: competitive in golden.yaml.
# expected_points is new and is for the hand rubric, not for the harness.
version: 1
questions:
  - id: comp_field_share
    job: meta
    warehouse: any
    question: >-
      Which decks held the biggest share of last week, and over how many games?
    expect_tools: [query_marts]
    require:
      - "re:\\b\\d+ games?\\b"
      - "re:share|most played|of the field"
    forbid: ["re:[0-9a-f]{16}", "deck inclusion"]
    expected_points:
      - names the archetypes in words, not by key
      - gives week_games, the per-game count, not the seat count
      - says share sums to about two because every game has two decks
      - flags min_games_met where it is false

  - id: comp_matchup_with_confidence
    job: meta
    warehouse: any
    question: >-
      How does Dragapult control do against the field, and how sure can I be?
    expect_tools: [query_marts]
    require:
      - Dragapult control
      - "re:\\b\\d+ games?\\b"
      - "re:thin|too few|not enough|below|unreliable|cannot be sure"
    forbid: ["re:[0-9a-f]{16}"]
    expected_points:
      - the record before the percentage
      - the sample size in the same sentence as the rate
      - an explicit statement that a rate over this many games is not a ranking

  - id: comp_bad_matchups_that_matter
    job: meta
    warehouse: any
    question: >-
      Which decks beat Dragapult control most often, and are those decks
      actually being played?
    expect_tools: [query_marts]
    require:
      - Dragapult control
      - "re:\\b\\d+ games?\\b"
    forbid: ["re:[0-9a-f]{16}"]
    expected_points:
      - reads the pairing rows and the weekly rows, and says which gave which
      - names every row tied on the top value rather than picking one
      - says plainly when a bad matchup has too few games to act on

  - id: comp_two_deck_choice
    job: meta
    warehouse: any
    question: >-
      I can bring one of two decks this weekend. Which is the safer choice?
    expect_tools: [query_marts]
    require:
      - "re:\\b\\d+ games?\\b"
      - "re:cannot|no row|does not hold|too few|thin"
    forbid: ["re:[0-9a-f]{16}"]
    expected_points:
      - gives both records with their sample sizes
      - says the warehouse holds no expected field to weight the matchups by
      - either declines to pick, with a reason, or picks and states how thin

  - id: comp_cards_to_expect
    job: meta
    warehouse: any
    question: What cards should I expect to see from Dragapult control?
    expect_tools: [query_marts]
    require:
      - "re:seen|observed"
      - "re:\\b\\d+ games?\\b"
    forbid: ["deck inclusion", "re:[0-9a-f]{16}"]
    expected_points:
      - reports the seen rate as an observation and a lower bound
      - gives the inclusion rate only where it is non-null, and says what it is over
      - never states a figure for how often a card is in the list

  - id: comp_my_record_by_side
    job: my_record
    warehouse: fixture
    context: "The member's player token is {player_token}."
    question: Do I win more going first, and is the difference worth anything?
    expect_tools: [query_marts]
    require:
      - "re:\\b\\d+ games?\\b"
      - "re:first|second"
    forbid: ["answer:re:[0-9a-f]{16}"]
    expected_points:
      - both halves, each with the games behind it
      - the record as a record before any percentage
      - says the split is too small to read, where it is

  - id: comp_my_worst_matchup
    job: my_record
    warehouse: fixture
    context: "The member's player token is {player_token}."
    question: Which deck have I lost to most?
    expect_tools: [query_marts]
    require:
      - "re:cannot|do not hold|no breakdown|not recorded|per opponent"
      - "re:\\b\\d+ games?\\b"
    forbid: ["answer:re:[0-9a-f]{16}"]
    expected_points:
      - says in one sentence that the member's record is not split by opponent
      - does not explain tokens or identity
      - offers the nearest thing it does hold, named, with its sample size

  - id: comp_my_week_trend
    job: my_record
    warehouse: fixture
    context: "The member's player token is {player_token}."
    question: How has my win rate moved week by week?
    expect_tools: [query_marts]
    require:
      - "re:cannot|do not hold|not by week|no weekly"
    forbid: ["answer:re:[0-9a-f]{16}"]
    expected_points:
      - names the limit precisely: the member row has no week grain
      - gives the overall record with its dates as the nearest answer
      - offers the community's weekly view as the thing that does exist

  - id: comp_post_loss_three_facts
    job: my_mistake
    warehouse: any
    question: What decided that game?
    expect_tools: []
    max_unverified: 0
    require:
      - "re:instead|could have|rather than|next time"
      - "re:matchup|no row|holds no|the league"
    forbid:
      - "re:they (probably|must have|likely) (had|held|drew)"
      - "re:[0-9a-f]{16}"
    expected_points:
      - three facts, cited, chosen for divergence rather than for size
      - one line at a named turn, written as a choice and not as a verdict
      - the matchup sentence, with its games count or an honest absence

  - id: comp_post_loss_conversion
    job: my_mistake
    warehouse: any
    question: >-
      My first attack was early and my first prize was late. Is that me or the
      matchup?
    expect_tools: []
    max_unverified: 0
    require:
      - "re:\\bturn \\d+"
      - "re:cannot say|both|no row|holds no|not enough"
    forbid:
      - "re:they (probably|must have|likely) (had|held|drew)"
      - "re:[0-9a-f]{16}"
    expected_points:
      - names the two turns and the gap between them
      - says plainly that nothing in the data separates pilot from pairing
      - does not pick one and present it as the finding

  - id: comp_pace_against_the_field
    job: my_game
    warehouse: any
    question: Was the pace of that game normal?
    expect_tools: [query_marts]
    max_unverified: 0
    require:
      - "re:\\bturn \\d+"
      - "re:average|usual|typically|on average"
      - "re:\\b\\d+ games?\\b"
    forbid: ["re:[0-9a-f]{16}"]
    expected_points:
      - distinguishes a number from the game on screen from a mart average
      - names which deck the average is for, and over how many seats
      - says when the member's own deck is not named in the context

  - id: comp_tournament_boundary
    job: meta
    warehouse: any
    question: What are people playing at tournaments right now?
    expect_tools: []
    require:
      - "re:league|uploaded|this warehouse|these games|community"
      - "re:instead|but|what I can|I do have"
    forbid: ["re:[0-9a-f]{16}"]
    expected_points:
      - says in one sentence that the corpus is uploaded league games, not events
      - does not answer from general knowledge
      - offers the nearest real answer, the week's most played decks, by name
```

Four of the twelve grade a refusal that stays useful (`comp_my_worst_matchup`,
`comp_my_week_trend`, `comp_two_deck_choice`, `comp_tournament_boundary`). That
is deliberate: twenty of the forty-six questions in section 1 end in a
refusal, so the quality of a refusal is a larger share of this product than the
quality of an answer.

### Feedback votes, by job

PLA-190 shipped the application half: a thumbs vote stored as a count with the
job and the gate beside it, and the question and answer text only when the
member ticks consent. PLA-201 is the unread half. The numbers to put on the
weekly run, all of them already derivable from what is stored:

| number | why |
| --- | --- |
| down-vote rate by job | the six jobs are six products; `my_mistake` and `meta` failing are different failures |
| down-vote rate by gate | a correlation with `allowed_low` is the gate costing answers, which is the `allowed_low` parked item in numbers |
| down-votes whose answer carried `unverified_numbers` greater than zero | whether members notice an untraceable number; if they do not, the check is for us and not for them |
| consented down-votes turned into golden cases per month | the only number that says the set is growing from use rather than from imagination |
| chips versus typed questions, by job | whether members ask their own questions or only the ones we wrote |

### One week, one or two players

Two members who play competitively, asked to use it for a week before a league
night and after a loss, with no other instruction. Read-only, no new storage,
nothing in this trial needs stored conversations.

What to look at:

| measure | source |
| --- | --- |
| questions per player per day, and when in the day | the service's `agent answered` log line |
| share typed rather than clicked from a chip | the same, with the offered strings known |
| share of answers that read at least one row | `gate_summary` and the evidence, already on the response |
| share of answers that are a refusal | the refusal pattern the adversarial set already uses |
| down-vote rate, and the notes from consented down-votes | PLA-190 rows, read by PLA-201 |
| median and worst latency | the service's own `latency_ms` |
| cost per player-day | `usage` on the `agent answered` line, priced at list |

What "better" means, written down before the week so it cannot be read to
taste:

1. the competitive subset scores at least 6 of 8 mean on the rubric, with
   "names the uncertainty" at 2 on at least four fifths of the cases;
2. all 70 offered chips answer on prod, with no refusal and at least one row
   read;
3. the down-vote rate is under one in ten, and no down-vote cites a number the
   member could not trace;
4. and the one that matters more than the other three: at least one of the two
   players changed a deck choice or a line because of an answer, and can say
   which answer.

If the first three pass and the fourth does not, the product is accurate and
nobody needs it, which is exactly the finding this spike exists to be able to
make.

---

## 6. Proposed tickets

Ordered by questions unlocked per unit of work. The top three are marked.

| id | title | size | unlocks |
| --- | --- | --- | --- |
| **T1** | **Per-member marts: the member's record at the grains they ask about** | M | Q10, Q25, Q37, Q39, Q40, Q41, Q42, Q44 |
| **T2** | **A refusal that names the nearest answerable question** | S | improves 20 refusals |
| **T3** | **Name the member's own deck in the game context, and use it** | S | Q30, Q33, Q34, Q41, and part 3 of every post-loss review |
| T4 | Confidence intervals beside every rate | S | repairs 16 weak answers |
| T5 | Widen the facts the request carries | S | Q27, Q29, Q34 |
| T6 | Turn-level marts: going first, pace by matchup, the prize map | M | Q09, Q18, Q19, Q31 |
| T7 | Decklist rows in silver, and a decklist mart | M | Q15, Q16, Q17 |
| T8 | Make every offered chip answerable, then make the check a gate | S | the 7 failing chips |
| T9 | Run the model comparison designed in section 3 | S | decides the routing question |
| T10 | Reconstructed-tier facts, with their caveats carried into the answer | M | Q29, Q31, Q34 |
| T11 | Licence and attribution check for an external meta source | S | unblocks or closes Q01, Q04, Q05, Q07, Q20, Q21 |
| T12 | Do not draw a chart from a number the rows do not hold | S | the `allowed_low` parked item |

### T1. Per-member marts: the member's record at the grains they ask about (top three)

**Why.** `mart_player_summary` is one row per member, and eight of the
forty-six questions a competitive player asks about themselves need a finer
grain: by deck, by opponent, by week, and by pace against the field. The
application is already offering three of those as chips and the agent is
refusing them. Every one of these marts is a group-by over a fact table that
already holds the columns; nothing new has to be ingested, parsed or stored.

**What.** Four dbt models over `fct_game_side` and `int_game_side_pace`:
`mart_player_archetype` (member by archetype by opening side),
`mart_player_matchup` (member by opponent archetype), `mart_player_weekly`
(member by ISO week), `mart_player_pace` (the ten pace numbers averaged over the
member's seats). Each carries `games` and `min_games_met` like every other mart.
Add them to `ALLOWED_TABLES`, regenerate `pipeline/marts_schema.py`, raise
`MAX_PROMPT_CHARS` for the four new schema blocks, and extend the `my_record`
playbook to say which of the five member tables answers which question.

**Done when.** The four models are built by `python -m pipeline.gold` and tested
by `dbt test` with `unique` and `not_null` on every key and a row-count test
against `fct_game_side`; `pipeline.marts_schema` is regenerated and its drift
test is green; `/health` reports twelve schema tables; the prompt test's
`MAX_PROMPT_CHARS` and `MIN_PREFIX_TOKENS` both pass; four new `warehouse: any`
golden cases, one per mart, pass against the fixture warehouse, including one
that asserts the thin-cell caveat is in words; and the three offered chips
"Which archetype have I lost to most this month", "How has my win rate moved
week by week" and "How does my pace compare to other players" answer on dev with
at least one row read and no refusal.

### T2. A refusal that names the nearest answerable question (top three)

**Why.** Twenty of the forty-six questions in this document end in a refusal,
and the refusals are accurate and useless. A member who asks which deck they
lose to most and is told the warehouse does not hold that breakdown has learned
nothing they can use; a member told the same sentence followed by "here is how
the three decks you have played most do against the field this month, over 41
games" has. This is the single change in this document with the best ratio of
member value to work, it is entirely in the prompt and the playbooks, and no
larger model will do it reliably without being asked to.

**What.** One sentence in each of the six playbooks saying what to do when the
question cannot be answered: name the limit in one sentence, in the member's
terms and never in the warehouse's, then answer the nearest question the tables
do cover and give that answer in full rather than offering it. A new `kind:
redirect` in the golden set whose `require` entries are the shape of a good
redirect (a limit named, a real number given, a sample size beside it) and whose
`forbid` entries are the shapes of a bad one (an explanation of tokens, an
apology of more than one sentence, a suggestion the member go and look
somewhere else). Four of the twelve proposed cases in section 5 are already
written as this kind.

**Done when.** Each of the six playbooks carries the sentence; at least six
`redirect` cases are in the golden set and pass, covering a member question with
no grain, a tournament question, a question about another player by name and a
matchup with no games; the broken-prompt control still fails them, which is what
shows the playbook sentence is doing the work rather than the model; and on the
`--offered` run no failing chip returns a bare refusal.

### T3. Name the member's own deck in the game context, and use it (top three)

**Why.** The page context for a member's own game names the opponent's
archetype and leaves the member's own deck as "your deck". That one missing
field is why the third part of a post-loss review, placing the game against how
this deck usually plays, cannot be written: `mart_archetype_pace` holds the
answer and the agent has no key to look it up with. The same omission weakens
four questions in section 1. It is one field on a request that already carries a
game summary, a fact list and a player token.

**What.** The application names the member's own archetype in the game summary
it sends, in the same form the opponent's is named. The `my_game` and
`my_mistake` playbooks gain a sentence: when the member's own archetype is
stated, read its pace row and say whether this game's first attack and first
prize are early or late for that deck, with the seats behind the average; when
it is not stated, say that in one short clause rather than comparing to nothing.
The fixtures under `evals/fixtures/facts/` are regenerated with the field
present, and the glossary gains nothing because the pace columns are already
defined there.

**Done when.** The fixture contexts name both archetypes; the
`job_my_mistake_review` and `job_my_game_walkthrough` golden cases are extended
to require the member's own deck named and a pace comparison with its seat
count, and pass; one new case with the field deliberately absent passes by
requiring the short clause and forbidding a comparison; and
`comp_pace_against_the_field` from section 5 passes on the fixture warehouse.

### The rest, in one line each

- **T4.** Wilson intervals on `mart_matchups` and `mart_archetype_weekly`, and
  a prompt rule putting the interval in the same sentence as the rate. Done when
  both marts carry `win_rate_low` and `win_rate_high`, the rule is in `RULES`,
  and two golden cases fail without the interval.
- **T5.** Widen the facts the request carries, one glossary line per new fact,
  regenerated fixtures, two new `mistake` cases. Done when the glossary test
  passes with the new ids and the new cases hold `max_unverified: 0`.
- **T6.** `mart_archetype_first_turn`, `mart_matchup_pace`, `mart_prize_map`.
  Done when each is built, tested, allowlisted and has a golden case, and the
  prompt still fits its ceiling.
- **T7.** A silver table at (game, seat, card) with a count, and
  `mart_decklist_cards` over it. Done when the modal count of a card in an
  archetype's lists is answerable and labelled as being over decklist-backed
  seats only. The contrast mart is a separate ticket and needs its own threshold.
- **T8.** Make the seven failing offered chips answerable, then turn the
  `--offered` step from a notice into a gate. Done when 70 of 70 pass on prod
  twice running and `continue-on-error` is removed.
- **T9.** Run section 3 as written, two repetitions, and record the rubric CSV
  and the priced report. Done when the three models' numbers are in one table
  and the routing rule in section 3 has been applied to them, whichever way it
  comes out.
- **T10.** Reconstructed-tier facts with caveats, mostly in the application.
  Done when at least two such facts arrive, each has a glossary line, the prompt
  rule carrying the caveat into the answer is in `RULES`, and a golden case fails
  when the caveat is dropped.
- **T11.** Read the external source's terms and write the four answers down.
  Done when the document says whether results may be stored, whether lists may
  be stored, what attribution is required and where, and whether a members-only
  product is in scope. No ingestion is designed before that page exists.
- **T12.** Charts draw only from returned rows. Done when a chart cannot be
  drawn from a number that is not in `evidence.queries[].rows`, and an answer
  with a low-confidence gate verdict and an untraceable number renders the
  caveat rather than the chart.

---

## What could not be judged from this repository

Six things, named so that nobody reads a judgement into their absence.

1. **The production corpus size.** The ticket says about 256 games and 175 full
   decklists; `README.md` says 128 and 67. Nothing in a checkout settles it, and
   `data/` was not read.
2. **Whether the deployed prompt is the one in this repository.** The service
   reports `prompt_sha256` and `schema_tables` on `/health` for exactly this
   comparison, and no call was made.
3. **Per-answer latency.** The harness records none, the only latency figures in
   the repository are emulator numbers for `/health`, and the service's own
   `latency_ms` is only reachable through `--remote`, which cannot be combined
   with `--model`.
4. **The `allowed_low` chart.** No chart of any kind is visible from this
   repository, and the member-facing documentation available to this spike
   describes an evidence table and no visualisation. T12 is written from the
   ticket's description of what was seen in a session, not from code.
5. **Whether an external meta source permits reuse.** No site was fetched. T11
   exists because the answer is unknown, not because it is expected to be yes.
6. **Whether the per-member grain in T1 crosses a line.** `docs/data-handling.md`
   reasons about the token, which is irreversible, and not about the grain. Four
   marts at member by week by archetype is a finer picture of one person than
   anything built so far, even keyed by a token, and that is the maintainer's
   call to make before the models are written.
