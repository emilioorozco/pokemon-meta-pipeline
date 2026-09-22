-- A placeholder dimension, and the description says so out loud.
--
-- There is no format column anywhere in the contract or in silver: the export
-- never names the ruleset a game was played under, so nothing upstream can be
-- keyed on it. The dimension is kept rather than dropped because the fact
-- needs a stable `format_key` slot for the day the producer starts recording
-- one, and a star schema that gains a dimension later is a bigger change than
-- one that gains a column. Until then the key is the export variant, which
-- describes the shape of the text the game was parsed from and not the game's
-- format at all, plus an `unknown` row for a game that names no variant.
with variants as (
    select distinct export_variant as format_key
    from {{ ref('stg_games') }}
    where export_variant is not null
)

select
    format_key,
    'Export variant: ' || format_key as format_name,
    false as is_real_format
from variants

union all

select
    'unknown' as format_key,
    'Unknown' as format_name,
    false as is_real_format
