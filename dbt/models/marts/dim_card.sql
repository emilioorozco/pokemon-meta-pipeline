-- One row per card seen anywhere in the corpus.
--
-- Built from `cards_seen` rather than from the card catalog on purpose. The
-- catalog is a 25k-entry export that is not committed here and is not always
-- fetched, and a dimension that fails to build without it would make the whole
-- gold stage depend on an optional file. Silver has already left joined the
-- catalog onto every observed card, so the catalog columns are here when the
-- file was present at silver time and null when it was not, and the dimension
-- builds either way. The cost is that a card nobody has played has no row,
-- which is the right size for a dimension the facts are the only reader of.
select
    card_key,
    max(card_name) as card_name,
    max(base_card_id) as base_card_id,
    max(set_code) as set_code,
    max(number) as number,
    max(catalog_name) as catalog_name,
    max(catalog_set) as catalog_set,
    max(catalog_type) as catalog_type,
    max(catalog_hp) as catalog_hp,
    max(catalog_reg) as catalog_reg,
    max(catalog_name) is not null as in_catalog,
    min(play_date) as first_seen,
    max(play_date) as last_seen
from {{ ref('stg_cards_seen') }}
group by card_key
