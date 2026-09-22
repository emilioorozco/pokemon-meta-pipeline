{{ config(materialized='ephemeral') }}

-- The seats the model is allowed to learn from, defined once.
--
-- Ephemeral on purpose: this is not a table anybody queries, it is the scope
-- rule, and the split cutoff and the feature table both need it. Compiling it
-- into each of them as a common table expression is what keeps the two from
-- disagreeing about which games are in scope, which would silently move the
-- train and holdout boundary.
--
-- Five exclusions, each for a reason that would otherwise turn into a wrong
-- number rather than a missing row:
--
-- - No archetype on either seat. A row whose deck is unknown cannot teach a
--   matchup, and bucketing the unknowns together would invent an archetype
--   that beats or loses to everything. Most uploader seats have no archetype
--   today (docs/stages.md, open items), so this is the exclusion that costs
--   the most rows, and it is the one upstream can fix.
-- - A result that is not a win or a loss. `tie` and `unknown` have no label:
--   a tie only happens on a hand-logged game and `unknown` means the seat
--   could not be resolved. Neither is a zero and neither is a one.
-- - No turns. A manual game is summary only, so there is no per-turn state to
--   describe and a feature row would be all zeros against a real label.
-- - `excluded_from_stats`. The uploader asked for the game to be left out of
--   the numbers, and every mart honours that; a model trained on it would put
--   it back in through the side door.
-- - An unresolved first player. `went_first` is a feature, and a null feature
--   is either a dropped row or a fabricated `false`. Dropping is the honest
--   one. On the current corpus this excludes nothing: every game that has a
--   log has a first player.
select
    f.game_id,
    f.seat,
    f.date_key as play_date,
    f.turn_count,
    f.went_first,
    f.is_uploader,
    f.archetype_key,
    f.opponent_archetype_key,
    f.is_win as won
from {{ ref('fct_game_side') }} as f
where not f.excluded_from_stats
  and f.archetype_key is not null
  and f.opponent_archetype_key is not null
  and f.result_for_seat in ('win', 'loss')
  and f.turn_count > 0
  and f.went_first is not null
