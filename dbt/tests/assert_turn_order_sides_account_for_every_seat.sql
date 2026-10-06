-- The two seats of a pairing add up to the pairing, and both of them exist.
--
-- `mart_archetype_turn_order` splits a pairing by who opened the game, which
-- is the one thing about it a reader cannot check against another mart:
-- `mart_matchups` holds the same games with the split collapsed, and it is
-- not read here on purpose, because two marts over one fact that agree with
-- each other can still both be wrong about it. So the count is taken from the
-- fact itself.
--
-- Three failures are caught and they are different mistakes. A side missing
-- is the cross join in the model having stopped producing both seats, which
-- would turn "0 games going first" back into a row nobody can see. A sum that
-- is short is a filter applied to one half only. A sum that is long is a seat
-- counted twice, which is what the all-opponents row would do if it were a
-- `group by` over the pair rows rather than a second pass over the fact.
--
-- The seats whose opening side the log never recorded are counted from the
-- fact rather than assumed to be zero, the way
-- `assert_player_summary_sides_account_for_every_seat` counts them: they
-- belong to neither half and a corpus that has some must still pass.
with fact_seats as (
    select
        archetype_key,
        opponent_archetype_key,
        went_first
    from {{ ref('fct_game_side') }}
    where not excluded_from_stats
      and archetype_key is not null
      and opponent_archetype_key is not null

    union all

    select
        archetype_key,
        'all' as opponent_archetype_key,
        went_first
    from {{ ref('fct_game_side') }}
    where not excluded_from_stats
      and archetype_key is not null
),

expected as (
    select
        archetype_key,
        opponent_archetype_key,
        count(*) as seats,
        count(*) filter (where went_first is null) as unknown_sides
    from fact_seats
    group by archetype_key, opponent_archetype_key
),

actual as (
    select
        archetype_key,
        opponent_archetype_key,
        count(*) as sides_present,
        coalesce(sum(games) filter (where went_first), 0) as games_first,
        coalesce(sum(games) filter (where not went_first), 0) as games_second
    from {{ ref('mart_archetype_turn_order') }}
    group by archetype_key, opponent_archetype_key
)

select
    e.archetype_key,
    e.opponent_archetype_key,
    e.seats,
    e.unknown_sides,
    a.sides_present,
    a.games_first,
    a.games_second
from expected as e
full outer join actual as a
    on a.archetype_key = e.archetype_key
   and a.opponent_archetype_key = e.opponent_archetype_key
where e.archetype_key is null
   or a.archetype_key is null
   or a.sides_present <> 2
   or a.games_first + a.games_second <> e.seats - e.unknown_sides
