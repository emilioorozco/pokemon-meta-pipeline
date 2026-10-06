-- A rate is between 0 and 1, and its interval is an interval around it.
--
-- The whole point of this mart is the pair of bounds beside the rate, so the
-- bounds are what a test has to hold. Four statements in one query, and each
-- of them is a different way the arithmetic could be wrong: a rate outside
-- the unit interval is a denominator that is not the decided games; a bound
-- outside it is the normal approximation having crept back in, which is the
-- one that goes negative at zero wins; a bound on the wrong side of the rate
-- is the half width subtracted from the wrong centre, which is the easy
-- mistake in Wilson because its centre is not the rate; and a rate with only
-- half an interval beside it is a null that leaked through one branch of the
-- arithmetic and not the other.
--
-- A row with nothing decided has a null rate and two null bounds and is not a
-- failure: it is a seat that has games and no result, which the corpus does
-- hold.
select
    turn_order_key,
    games,
    wins,
    losses,
    decided_games,
    win_rate,
    ci_low,
    ci_high
from {{ ref('mart_archetype_turn_order') }}
where (win_rate is not null and (win_rate < 0 or win_rate > 1))
   or (ci_low is not null and (ci_low < 0 or ci_low > 1))
   or (ci_high is not null and (ci_high < 0 or ci_high > 1))
   or (win_rate is not null and ci_low is not null and ci_low > win_rate)
   or (win_rate is not null and ci_high is not null and ci_high < win_rate)
   or ((win_rate is null) <> (ci_low is null))
   or ((ci_low is null) <> (ci_high is null))
