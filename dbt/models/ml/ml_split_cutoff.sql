{{ config(materialized='table', tags=['ml']) }}

-- One row, one date: where the training data stops and the holdout begins.
--
-- The split is by date and never random. Rows from the same game are near
-- duplicates of each other (a twenty turn game is twenty rows that share a
-- label), so a random split would put a game's turn 4 in train and its turn 5
-- in holdout and report a score that no future game will reproduce. A metagame
-- also moves: the question the model is asked is "this week's games, learned
-- from earlier weeks", and only a date split asks it.
--
-- The default cutoff is computed rather than typed, so the corpus growing does
-- not leave a stale date behind: of the distinct play dates that carry an
-- in-scope game, take the one where the games from that date to the end come
-- closest to `holdout_fraction` of all of them, preferring the later date when
-- two are equally close. It is a percentile over dates weighted by games, so
-- the boundary always falls between two days and never inside one.
--
-- `holdout_start` as a project variable overrides it, which is what a rerun of
-- an older experiment needs: `dbt run --vars '{holdout_start: 2026-09-01}'`.
with in_scope as (
    select distinct game_id, play_date
    from {{ ref('ml_labeled_side') }}
),

by_date as (
    select play_date, count(*) as games
    from in_scope
    group by play_date
),

ranked as (
    select
        play_date,
        sum(games) over (order by play_date desc) as games_from_here,
        sum(games) over () as games_total
    from by_date
),

chosen as (
    select
        play_date as computed_start,
        games_from_here as computed_holdout_games,
        games_total
    from ranked
    order by
        abs(games_from_here / cast(games_total as double) - {{ var('holdout_fraction') }}),
        play_date desc
    limit 1
)

select
    coalesce(
        try_cast(nullif('{{ var("holdout_start") }}', '') as date),
        computed_start
    ) as holdout_start,
    computed_start,
    computed_holdout_games,
    games_total
from chosen
