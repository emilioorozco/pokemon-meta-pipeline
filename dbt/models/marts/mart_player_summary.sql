-- One row per member: their record and the archetype they play most.
--
-- Members only, and that is not a filter choice made here: a stranger's token
-- is already NULL in silver (docs/data-handling.md), so there is no row to
-- summarize. Every number below is therefore about somebody who uploaded at
-- least one game and saw the notice.
--
-- The favourite archetype is the one with the most seats, ties broken by name
-- so a rebuild produces the same row twice.
--
-- The four going-first columns are the same record split by which side opened
-- the game. They are counted here rather than left to a query because "does
-- going first matter for me" is a question about one member, and the only
-- table keyed by a member is this one: without them the answer would need
-- `fct_game_side`, which is deliberately off the agent's allowlist. A seat
-- whose opening side the log never recorded is counted in neither half, so
-- the two add to `games` only for a member with no unknown side among their
-- seats; `assert_player_summary_sides_account_for_every_seat` is that
-- statement written out.
with scoped as (
    select *
    from {{ ref('fct_game_side') }}
    where not excluded_from_stats
      and player_key is not null
),

records as (
    select
        player_key,
        count(*) as games,
        count(*) filter (where is_uploader) as games_uploaded,
        count(*) filter (where is_win) as wins,
        count(*) filter (where is_loss) as losses,
        count(*) filter (where is_tie) as ties,
        count(*) filter (where went_first) as games_first,
        count(*) filter (where went_first and is_win) as wins_first,
        count(*) filter (where not went_first) as games_second,
        count(*) filter (where not went_first and is_win) as wins_second,
        min(date_key) as first_seen,
        max(date_key) as last_seen
    from scoped
    group by player_key
),

by_archetype as (
    select
        player_key,
        archetype_key,
        count(*) as games,
        row_number() over (
            partition by player_key
            order by count(*) desc, archetype_key
        ) as rank
    from scoped
    where archetype_key is not null
    group by player_key, archetype_key
)

select
    r.player_key,
    r.games,
    r.games_uploaded,
    r.wins,
    r.losses,
    r.ties,
    r.games_first,
    r.wins_first,
    r.games_second,
    r.wins_second,
    r.wins / nullif(cast(r.wins + r.losses as double), 0) as win_rate,
    f.archetype_key as favourite_archetype_key,
    a.archetype_name as favourite_archetype_name,
    f.games as favourite_archetype_games,
    r.first_seen,
    r.last_seen
from records as r
left join by_archetype as f on r.player_key = f.player_key and f.rank = 1
left join {{ ref('dim_archetype') }} as a on f.archetype_key = a.archetype_key
