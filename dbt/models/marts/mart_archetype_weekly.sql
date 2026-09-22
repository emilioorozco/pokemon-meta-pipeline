-- One row per archetype per International Organization for Standardization
-- (ISO) week: how often it was played and how it did.
--
-- Two different counts live in this model and mixing them up is the easy
-- mistake, so they are named apart.
--
-- `games` counts seat rows: every appearance of the archetype, on either side
-- of the table. That is the right numerator for a win rate, because a win
-- belongs to a seat.
--
-- `week_games` counts games, once each. It is taken from the uploader seats
-- only, because exactly one seat per game is the uploader, so counting those
-- rows counts each game exactly once with no `distinct` over the whole fact.
-- (A game with no resolved uploader seat is therefore not in the denominator;
-- it is also a game whose archetypes are mostly unknown, so it would only add
-- noise.)
--
-- `share_of_week` is `games / week_games`: the share of the week's games this
-- archetype was one of the two decks in. Because every game seats two decks,
-- the column sums to roughly two across a week, not one. That is deliberate,
-- and it is why the column is not called "metagame share": halving it would
-- make it sum to one but would stop it answering "how often did I face this".
with scoped as (
    select *
    from {{ ref('fct_game_side') }}
    where not excluded_from_stats
      and archetype_key is not null
),

weeks as (
    select
        date_key,
        iso_year,
        iso_week,
        week_start
    from {{ ref('dim_date') }}
),

week_totals as (
    select
        w.iso_year,
        w.iso_week,
        count(*) as week_games
    from {{ ref('fct_game_side') }} as f
    inner join weeks as w on f.date_key = w.date_key
    where not f.excluded_from_stats
      and f.is_uploader
    group by w.iso_year, w.iso_week
),

aggregated as (
    select
        s.archetype_key,
        w.iso_year,
        w.iso_week,
        w.week_start,
        count(*) as games,
        count(*) filter (where s.is_win) as wins,
        count(*) filter (where s.is_loss) as losses,
        count(*) filter (where s.is_tie) as ties
    from scoped as s
    inner join weeks as w on s.date_key = w.date_key
    group by s.archetype_key, w.iso_year, w.iso_week, w.week_start
)

select
    a.archetype_key || '-' || cast(a.iso_year as varchar) || '-W' || lpad(cast(a.iso_week as varchar), 2, '0')
        as archetype_week_key,
    a.archetype_key,
    d.archetype_name,
    a.iso_year,
    a.iso_week,
    a.week_start,
    a.games,
    a.wins,
    a.losses,
    a.ties,
    a.wins / nullif(cast(a.wins + a.losses as double), 0) as win_rate,
    t.week_games,
    a.games / nullif(cast(t.week_games as double), 0) as share_of_week,
    a.games >= {{ var('min_games') }} as min_games_met
from aggregated as a
left join week_totals as t on a.iso_year = t.iso_year and a.iso_week = t.iso_week
left join {{ ref('dim_archetype') }} as d on a.archetype_key = d.archetype_key
