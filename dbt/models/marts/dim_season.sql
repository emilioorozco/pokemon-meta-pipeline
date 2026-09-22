-- One row per season, plus a synthetic `unknown` row.
--
-- A season is read off the `[CompetitiveElo]` trailer, which only the debug
-- client prints, so most games have none. Rather than leave `season_key` null
-- on the fact and lose those rows to every inner join, the fact points at
-- `unknown` and the dimension carries a row for it. `is_known` is how a query
-- tells the placeholder apart from a real season.
with seasons as (
    select
        season_id as season_key,
        max(season_name) as season_name,
        min(play_date) as first_seen,
        max(play_date) as last_seen,
        count(*) as games
    from {{ ref('stg_games') }}
    where season_id is not null
    group by season_id
)

select
    season_key,
    season_name,
    true as is_known,
    first_seen,
    last_seen,
    games
from seasons

union all

select
    'unknown' as season_key,
    'Unknown season' as season_name,
    false as is_known,
    min(play_date) as first_seen,
    max(play_date) as last_seen,
    count(*) filter (where season_id is null) as games
from {{ ref('stg_games') }}
