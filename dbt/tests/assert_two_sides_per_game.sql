-- Every game has exactly two seats, so the fact has exactly two rows per game.
--
-- Silver reconciles the same invariant against bronze, and this repeats it on
-- the other side of the join: the fact joins seats to games, and a join that
-- lost or duplicated a game would pass silver's check and fail here.
select
    game_id,
    count(*) as sides
from {{ ref('fct_game_side') }}
group by game_id
having count(*) <> 2
