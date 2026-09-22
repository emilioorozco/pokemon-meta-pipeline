{{ config(materialized='table', tags=['ml']) }}

-- The training table for the win-probability model: one row per (game, seat,
-- turn number), describing the board as it was known at the START of that
-- turn, labelled with how the game ended.
--
-- Two decisions carry the whole model, and both are about leakage.
--
-- The first is the word "start". Every counter here is summed over turns
-- strictly before `turn_number`, so a row never contains the turn it is asked
-- about, and turn 1 is all zeros for both seats. The obvious implementation,
-- a running total that includes the current turn, would put the sixth prize on
-- the row of the turn that took it and the model would learn to read the
-- answer off the board.
--
-- The second is that both seats get a row at every turn number. A turn belongs
-- to one player, so `stg_turns` has one row per turn, and a naive join would
-- give a seat feature rows only on its own turns. Then `prizes_taken_opp`
-- would be read at a different moment from `prizes_taken_self`, and the two
-- seats of one game would not be comparable. Instead the seats are crossed
-- with the turn numbers 1..turn_count, and the seat that is waiting gets the
-- same snapshot the seat that is playing gets. `turns_played_self` is what
-- tells the model whose turn it is: at turn n the player who went first has
-- taken ceil(n/2) of their own turns and the other has taken floor(n/2).
--
-- Bench size, hand size and energy in play are NOT here, because the log
-- counters cannot produce them: `n_play_pokemon` counts the act of putting a
-- Pokemon down and nothing counts one leaving, so a cumulative sum of it is
-- "Pokemon played so far", not "Pokemon on the bench now". Guessing one from
-- the other would be a fabricated feature. See docs/features.md.
with sides as (
    select * from {{ ref('ml_labeled_side') }}
),

cutoff as (
    select holdout_start from {{ ref('ml_split_cutoff') }}
),

-- One row per (game, seat, turn), summed rather than assumed unique: a log
-- that splits one turn across two segments must not produce two rows here.
turn_counters as (
    select
        game_id,
        seat,
        turn_number,
        sum(n_draw) as n_draw,
        sum(n_attach) as n_attach,
        sum(n_attack) as n_attack,
        sum(n_play_pokemon) as n_play_pokemon,
        sum(n_play_trainer) as n_play_trainer,
        sum(n_evolve) as n_evolve,
        sum(n_knockout) as n_knockout,
        sum(n_prize_taken) as n_prize_taken
    from {{ ref('stg_turns') }}
    where turn_number is not null
      and seat is not null
    group by game_id, seat, turn_number
),

-- Both seats at every turn number the game reached. `generate_series` is
-- inclusive, and a game with `turn_count` 0 is already out of scope.
grid as (
    select
        s.game_id,
        s.seat,
        s.play_date,
        s.turn_count,
        s.went_first,
        s.is_uploader,
        s.archetype_key,
        s.opponent_archetype_key,
        s.won,
        unnest(generate_series(1, s.turn_count)) as turn_number
    from sides as s
),

-- The join condition is the leakage guard: `t.turn_number < g.turn_number`,
-- strictly less. The two `filter` clauses split the same scan into this seat's
-- history and the other seat's, so the snapshot is one pass over the turns.
state as (
    select
        g.game_id,
        g.seat,
        g.turn_number,
        coalesce(sum(t.n_prize_taken) filter (where t.seat = g.seat), 0) as prizes_taken_self,
        coalesce(sum(t.n_prize_taken) filter (where t.seat <> g.seat), 0) as prizes_taken_opp,
        coalesce(sum(t.n_knockout) filter (where t.seat = g.seat), 0) as knockouts_self,
        coalesce(sum(t.n_knockout) filter (where t.seat <> g.seat), 0) as knockouts_opp,
        coalesce(sum(t.n_draw) filter (where t.seat = g.seat), 0) as cards_drawn_self,
        coalesce(sum(t.n_attach) filter (where t.seat = g.seat), 0) as energy_attached_self,
        coalesce(sum(t.n_play_pokemon) filter (where t.seat = g.seat), 0) as pokemon_played_self,
        coalesce(sum(t.n_play_trainer) filter (where t.seat = g.seat), 0) as trainers_played_self,
        coalesce(sum(t.n_evolve) filter (where t.seat = g.seat), 0) as evolutions_self,
        coalesce(sum(t.n_attack) filter (where t.seat = g.seat), 0) as attacks_self,
        count(distinct t.turn_number) filter (where t.seat = g.seat) as turns_played_self
    from grid as g
    left join turn_counters as t
        on t.game_id = g.game_id
       and t.turn_number < g.turn_number
    group by g.game_id, g.seat, g.turn_number
)

select
    g.game_id || '-' || cast(g.seat as varchar) || '-' || cast(g.turn_number as varchar)
        as feature_key,
    g.game_id,
    cast(g.seat as integer) as seat,
    cast(g.turn_number as integer) as turn_number,
    g.play_date,
    g.turn_count,
    g.went_first,
    g.is_uploader,
    g.archetype_key,
    g.opponent_archetype_key,
    cast(s.prizes_taken_self as integer) as prizes_taken_self,
    cast(s.prizes_taken_opp as integer) as prizes_taken_opp,
    -- Six prizes a side, so remaining is the complement. `greatest` is a
    -- floor, not a correction: the counter comes from prize lines in the log
    -- and a re-logged or double-counted line must not produce a negative
    -- count that a model would read as a seventh prize.
    cast(greatest(0, 6 - s.prizes_taken_self) as integer) as prizes_remaining_self,
    cast(greatest(0, 6 - s.prizes_taken_opp) as integer) as prizes_remaining_opp,
    cast(s.prizes_taken_self - s.prizes_taken_opp as integer) as prize_diff,
    cast(s.knockouts_self as integer) as knockouts_self,
    cast(s.knockouts_opp as integer) as knockouts_opp,
    cast(s.cards_drawn_self as integer) as cards_drawn_self,
    cast(s.energy_attached_self as integer) as energy_attached_self,
    cast(s.pokemon_played_self as integer) as pokemon_played_self,
    cast(s.trainers_played_self as integer) as trainers_played_self,
    cast(s.evolutions_self as integer) as evolutions_self,
    cast(s.attacks_self as integer) as attacks_self,
    cast(s.turns_played_self as integer) as turns_played_self,
    g.won,
    -- Before the cutoff is training, the cutoff day itself and everything
    -- after it is holdout. One row in `ml_split_cutoff`, so the cross join
    -- attaches the same date to every row.
    case when g.play_date < c.holdout_start then 'train' else 'holdout' end as split
from grid as g
inner join state as s
    on s.game_id = g.game_id
   and s.seat = g.seat
   and s.turn_number = g.turn_number
cross join cutoff as c
