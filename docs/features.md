# Features

The model stage learns from one table, `features_turn`, built by dbt from the
gold fact and the silver turn counters. This document is what it holds, where
each column comes from, why it should matter, and the things it cannot say.

Everything here is at the grain **(game, seat, turn number)**, and every row is
the state of the game **at the start of that turn**. That is the whole design:
a row is a question ("this side, this board, this far in, who wins?") and the
label is the answer, and the two must not overlap.

```
silver.turns ------\
                    >--- features_turn (one row per game, seat, turn)
gold.fct_game_side -/            |
                                 +--- ml_split_cutoff (one date)
```

## 1. How a row is built

Three steps, in `dbt/models/ml/`:

1. `ml_labeled_side` picks the seats the model may learn from (section 5).
2. Those seats are crossed with the turn numbers `1..turn_count`, so **both**
   seats have a row at **every** turn, not only at their own turns. A turn
   belongs to one player, so `silver.turns` has one row per turn; joining that
   directly would give each seat features only on its own turns, and then
   `prizes_taken_self` and `prizes_taken_opp` would be read at different
   moments and the two seats of one game would not be comparable.
3. Each counter is summed over that seat's turns **strictly before** this turn
   number. Turn 1 is therefore all zeros for both seats, and no row contains
   the turn it is being asked about.

Step 3 is the leakage guard, and it is enforced by a dbt test
(`assert_features_turn_is_start_of_turn`) rather than trusted: change the
join's `<` to `<=` and the test fails on the first turn-1 row. A second test
(`assert_features_turn_covers_both_seats`) pins step 2, because a dropped seat
would silently halve every opponent-side feature.

## 2. The columns

Source column names are silver's unless noted. "Cumulative" always means
"summed over this seat's turns before this one".

### Identity and bookkeeping (not given to the model)

| Column | Definition | Source | Why it is here |
|---|---|---|---|
| `feature_key` | `game_id`-`seat`-`turn_number` | derived | The grain, so `unique` is testable |
| `game_id` | The game | `fct_game_side.game_id` | Tracing a prediction back, and checking that no game straddles the split |
| `seat` | 0 or 1 | `fct_game_side.seat` | Part of the grain. Not a feature: it is an index into the contract's `players` array, not a property of play |
| `play_date` | Day the game was played | `fct_game_side.date_key` | What `split` is computed from. Not a feature: the model would learn the calendar |
| `turn_count` | Turns in the whole game | `games.turn_count` | The future of this row, so not a feature. Kept so a test can assert `turn_number <= turn_count` |
| `split` | `train` or `holdout` | `ml_split_cutoff` | Section 4 |
| `won` | **The label.** This seat won | `game_sides.result_for_seat = 'win'` | Ties and unresolved results are excluded rather than encoded |

### Given to the model

| Column | Definition | Source | Why it should matter |
|---|---|---|---|
| `turn_number` | The turn this row is the start of | `turns.turn_number` | The base rate of winning is not flat over a game, and a feature that is informative on turn 12 is noise on turn 2 |
| `went_first` | This seat took the first turn | `game_sides.went_first` | The oldest known advantage in the game, and the only feature fully available before a card is played |
| `archetype_key` | This seat's deck | `fct_game_side.archetype_key` | The single most informative thing a pre-game predictor has |
| `opponent_archetype_key` | The other seat's deck | `fct_game_side.opponent_archetype_key` | With the column above it forms the matchup, the pairwise structure the whole metagame is described by |
| `prizes_taken_self` | Cumulative `n_prize_taken` | `turns.n_prize_taken` | Taking six prizes is one of the two ways to win, so this is the score line |
| `prizes_taken_opp` | The same for the other seat | `turns.n_prize_taken` | The other half of the score |
| `prize_diff` | `prizes_taken_self` minus `prizes_taken_opp` | derived | The race as one number, and the feature a human reads first |
| `knockouts_self` | Cumulative `n_knockout` | `turns.n_knockout` | Close to the prize count but not equal to it, because a two-prize Pokemon pays two prizes for one knockout; the gap carries how expensive the board is |
| `knockouts_opp` | The same for the other seat | `turns.n_knockout` | Same, from the other side |
| `cards_drawn_self` | Cumulative `n_draw`, opening hand and mulligan draws included | `turns.n_draw` | A proxy for whether the deck is running, since a stalled hand does not draw |
| `energy_attached_self` | Cumulative `n_attach` | `turns.n_attach` | Attachments are rationed to roughly one a turn, so this tracks whether an attacker got powered on schedule |
| `pokemon_played_self` | Cumulative `n_play_pokemon` | `turns.n_play_pokemon` | How much board this seat has committed. A count of an action, not a board state, which is why it is not a bench size (section 3) |
| `trainers_played_self` | Cumulative `n_play_trainer` (`play_card` and `play_stadium`) | `turns.n_play_trainer` | The clearest tempo signal in a format where a turn is mostly trainers |
| `evolutions_self` | Cumulative `n_evolve` | `turns.n_evolve` | Separates a deck that has set up its main attacker from one still playing basics |
| `attacks_self` | Cumulative `n_attack` | `turns.n_attack` | A seat that has not attacked by turn six is either the control deck or the one that is losing |
| `turns_played_self` | This seat's own turns before this one | `turns`, counted | What tells the model whose turn it is about to be, which `turn_number` alone cannot say, and the denominator for turning any counter above into a rate |

### In the table, withheld from the model

Three columns are built and tested like the rest and then left out of the
design matrix. The trainer logs the list as the `excluded_features` parameter
of every run, so the choice is part of the experiment record rather than a
comment nobody reads.

| Column | Why it is withheld |
|---|---|
| `prizes_remaining_self` | Exactly `6 - prizes_taken_self`, floored at zero. Perfectly collinear with a column already in the list, so it would only split one importance score into two. Kept because a served payload reads better with the number a player actually looks at |
| `prizes_remaining_opp` | The same, for the other seat |
| `is_uploader` | This seat belongs to the member who uploaded the game. It is a fact about who kept the log, not about the game, and on this corpus it is close to a giveaway: the uploader won every eligible holdout game and 87% of eligible training games. Trained with it, the model scored 0.99 area under the curve and had learned who pressed the upload button. It is also unavailable at prediction time by definition, because a prediction is asked before the game is uploaded. The column stays in the table so the bias stays measurable |

`seat` is withheld for the same kind of reason: it is an array index, and
`went_first` and `is_uploader` are the two real asymmetries between the seats.

## 3. What is not here, and cannot be

**Bench size.** Not derivable from the log counters, and not faked. The turn
table counts *actions*: `n_play_pokemon` counts a Pokemon being put into play,
and nothing counts one leaving it, whether knocked out, retreated into the
active spot, or returned to hand. A cumulative sum of it is "Pokemon played so
far", which is what `pokemon_played_self` is called, and it drifts further from
the bench with every knockout. The same argument rules out hand size, energy in
play, damage on the board, and which Pokemon is active. Getting any of them
would mean the producer emitting a board state per turn, not this pipeline
guessing one; it is an upstream change, and the note in
[stages.md](stages.md) is where it belongs.

**Damage.** `game_sides.stats_damage_dealt` exists at the game grain, but there
is no per-turn damage counter, so it cannot be cut off at the start of a turn
and is not in the table. Adding it as a game total would leak the ending.

**Deck contents.** `cards_seen` is observation, not a decklist
([schema.md](schema.md) section 8.4), and what a seat has revealed by turn 4 is
itself a function of how the game is going. It is a real feature source and a
careful one; it is not in this first version.

## 4. The train and holdout split

`split` is `train` for a game played before `ml_split_cutoff.holdout_start` and
`holdout` for one played on or after it. It is by date, never random, for a
reason that matters more here than in most projects: the rows of one game are
near duplicates of each other, twenty rows sharing one label and differing by a
turn, so a random split puts turn 4 in training and turn 5 in the holdout and
reports a score no future game can reproduce. A metagame also moves, and the
question worth answering is "this week's games, learned from earlier weeks".

The cutoff is computed rather than typed, so a growing corpus does not leave a
stale date behind. Of the distinct play dates that carry an in-scope game, take
the one where the games from that date to the end come closest to
`holdout_fraction` (0.25) of all of them, preferring the later date on a tie.
It is a percentile over dates weighted by games, so the boundary always falls
between two days and a game is never split across it. A dbt test asserts that:
no `game_id` has rows on both sides.

The `holdout_start` project variable overrides the computed date, which is what
reproducing an older run needs:

```bash
uv run dbt run --project-dir dbt --profiles-dir dbt \
  --select tag:ml --vars '{holdout_start: "2026-09-01"}'
```

## 5. What is excluded from the table

`ml_labeled_side` holds the scope rules. Each one drops rows that would
otherwise become a wrong number rather than a missing one.

| Rule | Reason |
|---|---|
| Either archetype is null | A seat whose deck is unknown cannot teach a matchup, and bucketing the unknowns together would invent an archetype that beats and loses to everything. **This is by far the largest exclusion**, because the application derives the opponent's archetype from the log but not the uploader's own (see the open item in [stages.md](stages.md)) |
| `result_for_seat` is not `win` or `loss` | A tie or an unresolved seat has no label. Neither is a zero and neither is a one |
| `turn_count` is 0 | A hand-logged game is summary only, so there is no per-turn state to describe and a feature row would be all zeros against a real label |
| `excluded_from_stats` | The uploader asked for the game to be left out of the numbers and every mart honours that; training on it would put it back in through the side door |
| `went_first` is null | It is a feature, and a null feature is either a dropped row or a fabricated `false`. On the current corpus this drops nothing |

## 6. Known limits

- **The corpus is small, and the usable part is much smaller than the whole.**
  128 games land in bronze; the archetype rule above leaves 28 of them in
  `features_turn`, which is 596 rows over 56 seats. Every number the model
  reports should be read next to those counts, and the honest description of
  the stage is that it demonstrates the loop.
- **The archetype keys are close to player identifiers here.** With 26
  distinct archetypes across 28 games, and with the same handful of members
  uploading, knowing the deck pair nearly identifies the game and therefore who
  logged it. That is why the holdout scores look high, for the baseline as much
  as for the model, and why the comparison between the two is more informative
  than either number alone. It resolves with more players, not with better
  hyperparameters.
- **Archetypes are name-keyed when they have no shared identifier.**
  `archetype_key` is the shared archetype row's id when a game carries one and
  `name:<lowercased canonical name>` when it does not
  ([schema.md](schema.md) section 8.5). Two spellings that upstream has not
  merged are two features, and the alias map handles a rename but not a merge.
- **No player identity feature, by design.** An opponent who never uploaded a
  game never saw the in-app notice, so silver writes their token as NULL and
  there is nothing to key on ([data-handling.md](data-handling.md)). A
  per-player skill feature would be the strongest feature in a corpus this
  size, and it is exactly the feature this pipeline has decided not to build.
- **The counters are the producer's, not the game engine's.** They come from
  parsing a printed battle log, so a line the parser does not recognise is a
  counter that does not move. `games.unparsed_count` is the column that says
  how often that happened.
- **One format, one patch window.** Every game is from a narrow date range on
  one game version. Nothing here transfers across a set release, and the date
  split is the only thing that would show it if it stopped.
