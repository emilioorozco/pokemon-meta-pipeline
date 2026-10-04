{{ config(materialized='ephemeral') }}

-- The ten pace numbers for one seat of one game: how fast that deck got going.
--
-- This is the second writing of a definition the application already has. When
-- a member uploads a game, `analyzeGame` in `@pra/shared` computes these same
-- ten numbers for that member's own seat and stores them beside the parsed
-- blob. They are written again here, in SQL over silver, because "how does my
-- pace compare" is a question about every seat in the league and not about one
-- game, and the nightly build is the only place that sees all of them.
--
-- Two writings of one definition drift, so they are held together by a test
-- rather than by a comment: `tests/test_gold.py` runs this model over the ten
-- committed fixture games and asserts, for both seats of each, that every one
-- of the ten numbers equals what the application's own function produced for
-- that seat (`evals/fixtures/pace/`).
--
-- Ephemeral, so it is a common table expression inside `mart_archetype_pace`
-- and not a relation of its own. The grain is the seat, which is where the
-- numbers are checkable; the mart is the average over the seats of an
-- archetype, and a reader who wants a seat has `fct_game_side` for everything
-- else about it.
--
-- Three exclusions carry the definitions, and they are the application's.
--
--   A seat's "counted turns" are the turn segments it owned, minus the turn
--   the game was conceded during: that turn was given up rather than spent.
--   For the attack share only, turn 1 of the player who went first comes out
--   as well, because the rules forbid the attack on it and allow the energy.
--   So the two shares have slightly different denominators on purpose.
--
--   A line's seat is the one the log credits it to, not the one whose turn it
--   was. `stg_turns` carries both attributions and they disagree: a Pokemon
--   can go down on its own owner's turn from a card effect, so the knockout
--   and the prizes taken for it belong to the other seat.
--
--   The turn numbers are the game's, which both seats share a clock for, so
--   "by turn 4" means the fourth turn of the game and not this seat's fourth.
with turn_rows as (
    select *
    from {{ ref('stg_turns') }}
    where seat is not null
      and turn_number is not null
),

sides as (
    select *
    from {{ ref('fct_game_side') }}
),

-- Game level: the turn a concession ended the game on, carried onto both
-- seats, which is how the application writes it. `max` rather than `min`
-- because the application keeps the last concede line it walks past, and a
-- log with two of them has not been seen.
concessions as (
    select
        game_id,
        max(turn_number) as concession_turn
    from turn_rows
    where concession
    group by game_id
),

-- The turn grain turned from "one row per segment, owned by one seat" into
-- "one row per seat per turn". The first half of the union is what the log
-- credits to the seat whose turn it was; the second half is what the same
-- turn credited to the other seat.
credited as (
    select
        game_id,
        seat,
        turn_number,
        n_attack_self as attacks,
        n_energy_attach_self as energy,
        n_prize_self as prizes,
        n_knockout_self as knockouts
    from turn_rows

    union all

    select
        game_id,
        1 - seat,
        turn_number,
        n_attack_opp,
        n_energy_attach_opp,
        n_prize_opp,
        n_knockout_opp
    from turn_rows
),

-- Summed rather than assumed unique, for the same reason `features_turn`
-- sums: a log that splits one turn across two segments must not make two rows.
by_turn as (
    select
        game_id,
        seat,
        turn_number,
        sum(attacks) as attacks,
        sum(energy) as energy,
        sum(prizes) as prizes,
        sum(knockouts) as knockouts
    from credited
    group by game_id, seat, turn_number
),

-- One row per turn this seat owned, with the two exclusions and whether it
-- attacked on it.
own_turns as (
    select
        t.game_id,
        t.seat,
        coalesce(c.concession_turn = t.turn_number, false) as conceded_on,
        coalesce(f.went_first, false) and t.turn_number = 1 as attack_forbidden,
        coalesce(k.attacks, 0) > 0 as attacked
    from turn_rows as t
    left join concessions as c on c.game_id = t.game_id
    left join sides as f on f.game_id = t.game_id and f.seat = t.seat
    left join by_turn as k
        on k.game_id = t.game_id
       and k.seat = t.seat
       and k.turn_number = t.turn_number
),

counted as (
    select
        game_id,
        seat,
        count(*) filter (where not conceded_on) as attach_turns,
        count(*) filter (where not conceded_on and not attack_forbidden) as attack_turns,
        count(*) filter (where not conceded_on and not attack_forbidden and not attacked)
            as turns_without_attack
    from own_turns
    group by game_id, seat
),

-- The whole game, with no exclusions: an energy attached on a turn that was
-- conceded during was still attached, and a first prize is a first prize.
totals as (
    select
        game_id,
        seat,
        sum(energy) as energy,
        min(turn_number) filter (where attacks > 0) as first_attack_turn,
        min(turn_number) filter (where prizes > 0) as first_prize_turn,
        min(turn_number) filter (where knockouts > 0) as first_knockout_turn,
        coalesce(sum(prizes) filter (where turn_number <= 4), 0) as prizes_by_turn_4,
        coalesce(sum(prizes) filter (where turn_number <= 6), 0) as prizes_by_turn_6,
        coalesce(sum(prizes) filter (where turn_number <= 8), 0) as prizes_by_turn_8,
        coalesce(sum(prizes) filter (where turn_number <= 10), 0) as prizes_by_turn_10
    from by_turn
    group by game_id, seat
)

select
    s.game_side_key,
    s.game_id,
    s.seat,
    s.archetype_key,
    s.excluded_from_stats,
    cast(c.attach_turns as integer) as counted_turns,
    cast(t.first_attack_turn as integer) as first_attack_turn,
    -- Null rather than zero when there was no turn the seat could have
    -- attacked on: a share of nothing is not a zero share.
    case
        when c.attack_turns > 0
            then round(cast(c.turns_without_attack as double) / c.attack_turns, 3)
    end as turns_without_attack_share,
    case
        when c.attach_turns > 0
            then round(coalesce(t.energy, 0) / cast(c.attach_turns as double), 3)
    end as energy_per_turn,
    cast(coalesce(t.prizes_by_turn_4, 0) as integer) as prizes_by_turn_4,
    cast(coalesce(t.prizes_by_turn_6, 0) as integer) as prizes_by_turn_6,
    cast(coalesce(t.prizes_by_turn_8, 0) as integer) as prizes_by_turn_8,
    cast(coalesce(t.prizes_by_turn_10, 0) as integer) as prizes_by_turn_10,
    cast(t.first_prize_turn as integer) as first_prize_turn,
    cast(t.first_knockout_turn as integer) as first_knockout_turn,
    cast(g.concession_turn as integer) as concession_turn
from sides as s
left join counted as c on c.game_id = s.game_id and c.seat = s.seat
left join totals as t on t.game_id = s.game_id and t.seat = s.seat
left join concessions as g on g.game_id = s.game_id
