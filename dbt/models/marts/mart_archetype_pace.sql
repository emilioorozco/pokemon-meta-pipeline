-- One row per archetype: how fast the deck plays, averaged over every seat
-- that played it.
--
-- The ten columns are the ten the application computes for a member's own
-- game at upload time, so a member's number and the community's number are
-- the same measurement and can be put side by side. `int_game_side_pace` is
-- where they are defined, one seat at a time, with the exclusions written
-- out; this model is the average and nothing else.
--
-- Every average skips the seats where its own number is null, which is not
-- the same set for every column: a seat that never attacked has no first
-- attack turn, a seat whose game ended on prizes has no concession turn. So
-- `games` is the seats the archetype played, not the denominator of any one
-- column, and a column over a small corpus can rest on fewer seats than
-- `games` says. `min_games_met` is the same flag the other marts carry.
--
-- A seat with no turn-by-turn log at all is out: `counted_turns` is null for
-- a hand-logged game, and averaging an archetype's pace over a game nobody
-- logged would make `games` a count of rows rather than of evidence.
with scoped as (
    select *
    from {{ ref('int_game_side_pace') }}
    where not excluded_from_stats
      and archetype_key is not null
      and counted_turns is not null
),

aggregated as (
    select
        archetype_key,
        count(*) as games,
        avg(first_attack_turn) as first_attack_turn,
        avg(turns_without_attack_share) as turns_without_attack_share,
        avg(energy_per_turn) as energy_per_turn,
        avg(prizes_by_turn_4) as prizes_by_turn_4,
        avg(prizes_by_turn_6) as prizes_by_turn_6,
        avg(prizes_by_turn_8) as prizes_by_turn_8,
        avg(prizes_by_turn_10) as prizes_by_turn_10,
        avg(first_prize_turn) as first_prize_turn,
        avg(first_knockout_turn) as first_knockout_turn,
        avg(concession_turn) as concession_turn
    from scoped
    group by archetype_key
)

select
    a.archetype_key,
    d.archetype_name,
    a.games,
    a.first_attack_turn,
    a.turns_without_attack_share,
    a.energy_per_turn,
    a.prizes_by_turn_4,
    a.prizes_by_turn_6,
    a.prizes_by_turn_8,
    a.prizes_by_turn_10,
    a.first_prize_turn,
    a.first_knockout_turn,
    a.concession_turn,
    a.games >= {{ var('min_games') }} as min_games_met
from aggregated as a
left join {{ ref('dim_archetype') }} as d on a.archetype_key = d.archetype_key
