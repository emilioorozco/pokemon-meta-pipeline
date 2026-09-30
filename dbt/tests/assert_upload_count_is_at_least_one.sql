-- `upload_count` is a count of the bronze rows a game was collapsed from, so a
-- game that is here at all was uploaded at least once and the column can never
-- be zero or negative. A range test rather than a `not_null`, which the schema
-- already carries: this is a singular test because the project installs no test
-- packages, so there is no `accepted_range` to reach for.
--
-- Both models that carry the column are checked, because they are two chances
-- to lose it: staging reads it off silver, and the fact joins it on.
select
    'stg_games' as model,
    game_id,
    upload_count
from {{ ref('stg_games') }}
where upload_count < 1

union all

select
    'fct_game_side' as model,
    game_id,
    upload_count
from {{ ref('fct_game_side') }}
where upload_count < 1
