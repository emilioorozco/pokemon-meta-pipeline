-- Two rows per game, one per seat. Renames and casts, plus the two surrogate
-- keys that every mart would otherwise re-derive: `archetype_key` and
-- `player_key`. They are defined once here so the dimension and the fact
-- cannot drift apart.
--
-- `archetype_key` is the shared archetype row's id when the game carries one,
-- and otherwise the canonical name lowercased and prefixed with `name:`. The
-- prefix keeps the two key spaces apart: an id is an uppercase 26-character
-- identifier, so a name key can never collide with one, and a reader can see
-- at a glance which archetypes have no shared row behind them yet.
--
-- The `stats_*` list columns (pokemon played, cards played, evolutions,
-- attacks) are dropped: nothing in gold aggregates them, and a list column
-- would travel through every downstream table for no reader.
select
    game_id,
    cast(play_date as date) as play_date,
    cast(seat as integer) as seat,
    coalesce(is_uploader, false) as is_uploader,
    -- Already NULL for a stranger: silver applies the rule
    -- (docs/data-handling.md), gold never sees a non-member token.
    player_token as player_key,
    coalesce(is_member, false) as is_member,
    archetype_id,
    archetype_name,
    archetype_name_raw,
    archetype_source,
    case
        when archetype_id is not null then archetype_id
        when archetype_name is not null then 'name:' || lower(trim(archetype_name))
    end as archetype_key,
    result_for_seat,
    went_first,
    cast(stats_cards_drawn as integer) as cards_drawn,
    cast(stats_energy_attached as integer) as energy_attached,
    cast(stats_damage_dealt as integer) as damage_dealt,
    cast(stats_knockouts as integer) as knockouts,
    cast(stats_prizes_taken as integer) as prizes_taken,
    cast(stats_mulligans as integer) as mulligans,
    cast(stats_turns_taken as integer) as turns_taken,
    decklist_source,
    decklist_complete,
    cast(decklist_card_count as integer) as decklist_card_count,
    -- The uploader's own deck record, passed through for player-level views.
    -- It is a nickname typed into the game client, not an archetype, and no
    -- archetype column above reads it.
    deck_name,
    deck_id
from {{ source('silver', 'game_sides') }}
