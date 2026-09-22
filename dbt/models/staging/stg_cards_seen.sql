-- One row per (game, seat, card) observed. Renames and casts, plus
-- `card_key`, which is just silver's `card_id`: that column is already the
-- resolvable identity (client card id, else base card id, else the lowercased
-- printed name) and is never null, so the fact needs no second coalesce.
--
-- `card_name` prefers the catalog's spelling over the battle log's, because
-- the log prints a card the way that game printed it and the catalog is the
-- one place a card has a single name.
select
    game_id,
    cast(play_date as date) as play_date,
    cast(seat as integer) as seat,
    card_id as card_key,
    base_card_id,
    coalesce(catalog_name, card_name) as card_name,
    card_name as observed_name,
    set_code,
    number,
    cast(count_seen as integer) as count_seen,
    in_decklist,
    catalog_name,
    catalog_set,
    catalog_type,
    cast(catalog_hp as integer) as catalog_hp,
    catalog_reg
from {{ source('silver', 'cards_seen') }}
