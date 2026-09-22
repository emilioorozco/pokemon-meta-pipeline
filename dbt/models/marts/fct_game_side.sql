-- The fact table: one row per (game, seat), two rows per game.
--
-- The seat grain is what makes every question in the marts a group-by rather
-- than a pivot: a win rate is `avg(is_win)`, a matchup is a group-by on the
-- two archetype keys, and a player's record is a group-by on `player_key`. The
-- opponent's archetype is denormalized onto the row for the same reason, so a
-- matchup query never has to self-join the fact.
--
-- Nothing is filtered here. `excluded_from_stats` is carried as a column and
-- every mart drops it; a fact that had already dropped those rows could not
-- answer "how many games were excluded", which is a real data-quality
-- question.
with sides as (
    select * from {{ ref('stg_game_sides') }}
),

games as (
    select * from {{ ref('stg_games') }}
),

-- The other seat of the same game. An inner join is wrong here: a game whose
-- other seat somehow went missing should still produce its own row with a null
-- opponent, and `assert_two_sides_per_game` is what catches that case loudly.
opponents as (
    select
        game_id,
        seat,
        archetype_key
    from sides
)

select
    s.game_id || '-' || cast(s.seat as varchar) as game_side_key,
    s.game_id,
    s.seat,
    s.is_uploader,
    s.player_key,
    s.archetype_key,
    o.archetype_key as opponent_archetype_key,
    coalesce(g.season_id, 'unknown') as season_key,
    coalesce(g.export_variant, 'unknown') as format_key,
    g.play_date as date_key,
    s.result_for_seat,
    s.result_for_seat = 'win' as is_win,
    s.result_for_seat = 'loss' as is_loss,
    s.result_for_seat = 'tie' as is_tie,
    s.went_first,
    g.turn_count,
    s.prizes_taken,
    s.knockouts,
    s.cards_drawn,
    s.energy_attached,
    s.damage_dealt,
    s.mulligans,
    s.turns_taken,
    s.decklist_complete,
    s.decklist_card_count,
    g.has_full_decklists,
    g.export_variant,
    g.excluded_from_stats
from sides as s
inner join games as g on s.game_id = g.game_id
left join opponents as o on s.game_id = o.game_id and s.seat <> o.seat
