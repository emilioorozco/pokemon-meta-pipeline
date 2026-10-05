-- `games_first` and `games_second` are a partition of `games`, less the seats
-- whose opening side the log never recorded.
--
-- The split is counted in the mart rather than derived by a reader, so this is
-- the statement that keeps it honest: a filter applied to one half only, or a
-- `went_first` that started arriving as a string rather than a boolean, would
-- leave the two halves adding to something other than the record they came
-- from. The unknown seats are counted from the fact rather than assumed to be
-- zero, so the test holds on a corpus where some games never said who opened.
with unknown_sides as (
    select
        player_key,
        count(*) as sides
    from {{ ref('fct_game_side') }}
    where not excluded_from_stats
      and player_key is not null
      and went_first is null
    group by player_key
)

select
    s.player_key,
    s.games,
    s.games_first,
    s.games_second,
    coalesce(u.sides, 0) as unknown_sides
from {{ ref('mart_player_summary') }} as s
left join unknown_sides as u on s.player_key = u.player_key
where s.games_first + s.games_second <> s.games - coalesce(u.sides, 0)
