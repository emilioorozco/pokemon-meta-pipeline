-- A vs B and B vs A describe the same set of games, so their game counts have
-- to match and the mirrored row has to exist at all.
--
-- The mart gets this for free from the seat grain rather than from a union, so
-- the test is really checking the grain: a fact row that lost its opponent's
-- archetype, or a filter applied to one side of the pair only, shows up here.
-- The wins of one side must also be the losses of the other, which is the
-- second condition.
with pairs as (
    select
        archetype_key,
        opponent_archetype_key,
        games,
        wins,
        losses
    from {{ ref('mart_matchups') }}
)

select
    a.archetype_key,
    a.opponent_archetype_key,
    a.games,
    b.games as mirrored_games,
    a.wins,
    b.losses as mirrored_losses
from pairs as a
left join pairs as b
    on a.archetype_key = b.opponent_archetype_key
   and a.opponent_archetype_key = b.archetype_key
where b.archetype_key is null
   or a.games <> b.games
   or a.wins <> b.losses
