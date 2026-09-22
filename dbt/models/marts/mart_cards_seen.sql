-- One row per (archetype, card): how often the card turned up when that
-- archetype was at the table.
--
-- Two rates sit side by side here and they measure different things. Read the
-- model description in schema.yml before using either.
--
-- `seen_rate` is over every in-scope seat the archetype held. `inclusion_rate`
-- is over the subset of those seats that shared a full decklist in game, which
-- is the only subset where "was this card in the deck" has an answer at all.
with seats as (
    select
        game_id,
        seat,
        archetype_key,
        decklist_card_count is not null as has_decklist
    from {{ ref('fct_game_side') }}
    where not excluded_from_stats
      and archetype_key is not null
),

denominators as (
    select
        archetype_key,
        count(*) as games,
        count(*) filter (where has_decklist) as decklist_games
    from seats
    group by archetype_key
),

observed as (
    select
        s.archetype_key,
        c.card_key,
        count(*) as games_with_card,
        count(*) filter (where s.has_decklist and c.in_decklist) as decklist_games_with_card,
        sum(c.count_seen) as copies_seen,
        max(c.count_seen) as max_copies_seen
    from seats as s
    inner join {{ ref('stg_cards_seen') }} as c
        on s.game_id = c.game_id and s.seat = c.seat
    group by s.archetype_key, c.card_key
)

select
    o.archetype_key || '|' || o.card_key as archetype_card_key,
    o.archetype_key,
    a.archetype_name,
    o.card_key,
    card.card_name,
    card.catalog_type,
    d.games,
    o.games_with_card,
    o.games_with_card / cast(d.games as double) as seen_rate,
    d.decklist_games,
    o.decklist_games_with_card,
    o.decklist_games_with_card / nullif(cast(d.decklist_games as double), 0) as inclusion_rate,
    o.copies_seen / cast(o.games_with_card as double) as avg_copies_seen,
    o.max_copies_seen
from observed as o
inner join denominators as d on o.archetype_key = d.archetype_key
left join {{ ref('dim_archetype') }} as a on o.archetype_key = a.archetype_key
left join {{ ref('dim_card') }} as card on o.card_key = card.card_key
