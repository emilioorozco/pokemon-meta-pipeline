-- One row per turn segment. Renames and casts only. Nothing in gold reads the
-- turn grain yet; it is staged so the win-probability model's turn-indexed
-- features have a typed view to start from rather than a raw Parquet glob.
select
    game_id,
    cast(play_date as date) as play_date,
    cast(turn_number as integer) as turn_number,
    cast(seat as integer) as seat,
    cast(n_entries as integer) as n_entries,
    cast(n_draw as integer) as n_draw,
    cast(n_attach as integer) as n_attach,
    cast(n_attack as integer) as n_attack,
    cast(n_play_pokemon as integer) as n_play_pokemon,
    cast(n_play_trainer as integer) as n_play_trainer,
    cast(n_evolve as integer) as n_evolve,
    cast(n_retreat as integer) as n_retreat,
    cast(n_knockout as integer) as n_knockout,
    cast(n_prize_taken as integer) as n_prize_taken,
    coalesce(concession, false) as concession
from {{ source('silver', 'turns') }}
