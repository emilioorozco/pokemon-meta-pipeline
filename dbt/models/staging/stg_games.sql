-- One row per game, straight off silver.games. Renames and casts only: the
-- nulls silver preserved on purpose stay null, except the two flags that are
-- read as false everywhere downstream and are coalesced once, here.
select
    game_id,
    user_id,
    cast(play_date as date) as play_date,
    cast(played_at as timestamp) as played_at,
    export_variant,
    upload_source,
    cast(turn_count as integer) as turn_count,
    end_reason,
    result,
    cast(winner_seat as integer) as winner_seat,
    cast(went_first_seat as integer) as went_first_seat,
    cast(first_player as integer) as first_player,
    cast(my_side as integer) as my_side,
    coalesce(excluded_from_stats, false) as excluded_from_stats,
    coalesce(has_full_decklists, false) as has_full_decklists,
    season_id,
    season_name,
    cast(ingested_at as timestamp) as ingested_at
from {{ source('silver', 'games') }}
