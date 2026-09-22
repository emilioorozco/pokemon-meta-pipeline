-- One row per member: somebody who uploaded at least one game and therefore
-- saw the in-app notice. Strangers are absent by construction, not by a filter
-- here: silver writes their token as NULL (docs/data-handling.md), so the
-- `where` below is only saying "the rows that still have a token".
--
-- `games_played` counts every seat the token holds, not only the uploader
-- seats, so a member who was matched against another member is counted on
-- both sides of that game.
select
    player_key,
    min(play_date) as first_seen,
    max(play_date) as last_seen,
    count(*) as games_played,
    count(*) filter (where is_uploader) as games_uploaded
from {{ ref('stg_game_sides') }}
where player_key is not null
group by player_key
