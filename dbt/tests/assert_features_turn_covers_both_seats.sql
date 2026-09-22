-- Every in-scope game contributes exactly two rows per turn, one per seat.
--
-- The scope rules are per seat but they always decide both seats the same way:
-- the two archetype keys are the same pair swapped, the two results mirror
-- each other, and the rest are properties of the game. So a game that appears
-- at all must appear twice at every turn number, and a game that appears once
-- means a seat was dropped by an accident rather than by a rule, which is the
-- failure that would quietly halve the opponent's side of every feature.
--
-- It also pins the turn grid: `turn_count` turn numbers, no gaps, because a
-- missing turn number would make the count come up short.
select
    game_id,
    count(*) as rows_built,
    2 * max(turn_count) as rows_expected
from {{ ref('features_turn') }}
group by game_id
having count(*) <> 2 * max(turn_count)
    or count(distinct seat) <> 2
    or count(distinct turn_number) <> max(turn_count)
