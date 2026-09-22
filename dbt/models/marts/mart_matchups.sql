-- Archetype A against archetype B: one row per ordered pair.
--
-- Symmetric by construction rather than by a union. A game puts two rows in
-- the fact, one per seat, and each of them carries its own archetype and the
-- other seat's, so the same game lands once as (A, B) and once as (B, A). An
-- application looking up either direction finds a row, and
-- `assert_matchups_symmetric` checks that the game counts agree.
--
-- A mirror, (A, A), is the one asymmetry worth knowing about: both seats of
-- the game produce the same ordered pair, so a mirror row counts each game
-- twice and its wins and losses are equal by definition.
with scoped as (
    select *
    from {{ ref('fct_game_side') }}
    where not excluded_from_stats
      and archetype_key is not null
      and opponent_archetype_key is not null
),

aggregated as (
    select
        archetype_key,
        opponent_archetype_key,
        count(*) as games,
        count(*) filter (where is_win) as wins,
        count(*) filter (where is_loss) as losses,
        count(*) filter (where is_tie) as ties,
        count(*) filter (where result_for_seat = 'unknown') as undecided,
        min(date_key) as first_played,
        max(date_key) as last_played
    from scoped
    group by archetype_key, opponent_archetype_key
)

select
    a.archetype_key || ' vs ' || a.opponent_archetype_key as matchup_key,
    a.archetype_key,
    mine.archetype_name,
    a.opponent_archetype_key,
    theirs.archetype_name as opponent_archetype_name,
    a.archetype_key = a.opponent_archetype_key as is_mirror,
    a.games,
    a.wins,
    a.losses,
    a.ties,
    a.undecided,
    -- Ties and unknown results are excluded from the denominator rather than
    -- counted as half a win: a tie only happens on a hand-logged game, and
    -- `unknown` means the seat could not be resolved, so neither is evidence
    -- about the matchup.
    a.wins / nullif(cast(a.wins + a.losses as double), 0) as win_rate,
    a.games >= {{ var('min_games') }} as min_games_met,
    a.first_played,
    a.last_played
from aggregated as a
left join {{ ref('dim_archetype') }} as mine on a.archetype_key = mine.archetype_key
left join {{ ref('dim_archetype') }} as theirs on a.opponent_archetype_key = theirs.archetype_key
