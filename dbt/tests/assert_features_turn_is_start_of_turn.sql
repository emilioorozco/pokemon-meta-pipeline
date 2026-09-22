-- The leakage guard, as a test rather than as a comment.
--
-- Three things have to hold if `features_turn` really is the state at the
-- start of a turn, and each of them breaks in a different way if the join in
-- that model loses its strict inequality:
--
-- - Turn 1 is all zeros. Nothing has happened yet, for either seat. Change
--   `t.turn_number < g.turn_number` to `<=` and this is the first row that
--   tells you.
-- - The turn is inside the game. A turn number above `turn_count` would mean
--   the grid outran the log.
-- - A seat cannot have taken more of its own turns than the turns that have
--   gone by.
select
    feature_key,
    'turn 1 is not empty' as failure
from {{ ref('features_turn') }}
where turn_number = 1
  and (
      prizes_taken_self <> 0
      or prizes_taken_opp <> 0
      or knockouts_self <> 0
      or knockouts_opp <> 0
      or cards_drawn_self <> 0
      or energy_attached_self <> 0
      or pokemon_played_self <> 0
      or trainers_played_self <> 0
      or evolutions_self <> 0
      or attacks_self <> 0
      or turns_played_self <> 0
  )

union all

select
    feature_key,
    'turn number outside the game' as failure
from {{ ref('features_turn') }}
where turn_number < 1 or turn_number > turn_count

union all

select
    feature_key,
    'more own turns than turns elapsed' as failure
from {{ ref('features_turn') }}
where turns_played_self > turn_number - 1
