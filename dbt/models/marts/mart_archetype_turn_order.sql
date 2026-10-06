-- Going first against going second: one row per (archetype, opponent, seat).
--
-- "Should I go first or second with this deck" has no public answer for the
-- paper rules game, because the sites that publish matchup tables do not
-- record who opened. This project does: `went_first` is a column of every
-- seat row in the fact, and this is the mart that reads it at the grain the
-- question is asked at.
--
-- The grain is three keys and not two. A pairing is played from both seats,
-- so the row a reader needs is the pair plus the side of the table, and a
-- mart keyed by the pair alone could only ever give the average of the two.
--
-- The all-opponents row is the same count with the opponent dropped, carried
-- here under the literal `all` rather than left to a reader's `group by`,
-- for the reason `mart_archetype_weekly` carries `week_games`: a sum over the
-- pair rows is one `where` clause away from double counting, and the question
-- "with this deck, in general" is the one most members are actually asking.
-- `all` is a literal and not a null for the reason `dim_season` carries an
-- `unknown` row: a key that is present is joinable and groupable, and a null
-- is lost to every inner join. It cannot collide with a real archetype, whose
-- key is a 26 character identifier or a `name:` prefixed label, and
-- `is_all_opponents` is how a query tells the two apart without matching on a
-- string.
--
-- Both seats are always present. A pairing played only from one side gets a
-- row of zeros on the other, because "0 games going first" is an answer and a
-- missing row is not: a mart that answers this question by omission invites
-- exactly the reading it exists to prevent, which is one seat's rate reported
-- as the deck's.
--
-- The all-opponents row keeps the seats whose opponent named no archetype,
-- which the pair rows cannot hold at all. Six of the ten fixture games are
-- such seats, and a deck's own first-or-second record does not stop being
-- its record because the other side of the table went unlabelled.
--
-- A seat whose opening side the log never recorded is in neither half, the
-- way `mart_player_summary` counts it, and
-- `assert_turn_order_sides_account_for_every_seat` is that statement written
-- out against the fact.
--
-- Scoped like every other mart in this directory and not by season: nothing
-- here filters on `season_key`, because no mart does. The season trailer only
-- exists in a debug export, so a season filter would drop most of the corpus,
-- and `fct_game_side` carries `season_key` for the day that changes.
with scoped as (
    select *
    from {{ ref('fct_game_side') }}
    where not excluded_from_stats
      and archetype_key is not null
      and went_first is not null
),

-- The pair rows and the all-opponents rows in one relation, so everything
-- below is written once.
seats as (
    select
        archetype_key,
        opponent_archetype_key,
        went_first,
        result_for_seat
    from scoped
    where opponent_archetype_key is not null

    union all

    select
        archetype_key,
        'all' as opponent_archetype_key,
        went_first,
        result_for_seat
    from scoped
),

aggregated as (
    select
        archetype_key,
        opponent_archetype_key,
        went_first,
        count(*) as games,
        count(*) filter (where result_for_seat = 'win') as wins,
        count(*) filter (where result_for_seat = 'loss') as losses
    from seats
    group by archetype_key, opponent_archetype_key, went_first
),

-- Every pairing that has a game, crossed with the two sides of the table.
grid as (
    select
        pairings.archetype_key,
        pairings.opponent_archetype_key,
        sides.went_first
    from (select distinct archetype_key, opponent_archetype_key from seats) as pairings
    cross join (select true as went_first union all select false) as sides
),

counted as (
    select
        g.archetype_key,
        g.opponent_archetype_key,
        g.went_first,
        coalesce(a.games, 0) as games,
        coalesce(a.wins, 0) as wins,
        coalesce(a.losses, 0) as losses,
        coalesce(a.wins, 0) + coalesce(a.losses, 0) as decided_games
    from grid as g
    left join aggregated as a
        on a.archetype_key = g.archetype_key
       and a.opponent_archetype_key = g.opponent_archetype_key
       and a.went_first = g.went_first
),

-- The denominator of the rate and of the interval, which is the decided games
-- and not `games`: ties and unresolved seats are left out here for the reason
-- `mart_matchups` leaves them out, because neither is evidence either way.
-- Null rather than zero, so a seat with nothing decided carries a null rate
-- and a null interval instead of a division error.
rated as (
    select
        c.archetype_key,
        c.opponent_archetype_key,
        c.went_first,
        c.games,
        c.wins,
        c.losses,
        c.decided_games,
        nullif(cast(c.decided_games as double), 0) as n,
        c.wins / nullif(cast(c.decided_games as double), 0) as p,
        -- The standard normal deviate for a two-sided 95% interval.
        1.959964 as z
    from counted as c
),

-- The Wilson score interval, and not wins over games plus or minus two
-- standard errors. The normal approximation is the one every spreadsheet
-- reaches for and it is wrong in exactly the place this mart lives: at four
-- games and four wins it gives an interval of zero width around 100%, and at
-- zero wins it gives one around 0%. Wilson is the interval of the rates that
-- would not have been rejected by what was seen, so it stays inside 0 and 1,
-- it is wide when the sample is small, and it never collapses on a sweep.
--
--     centre = (p + z^2 / 2n) / (1 + z^2 / n)
--     half   = z / (1 + z^2 / n) * sqrt(p(1 - p) / n + z^2 / 4n^2)
--
-- with p the win rate over the decided games, n those games and z the deviate
-- above.
bounded as (
    select
        r.archetype_key,
        r.opponent_archetype_key,
        r.went_first,
        r.games,
        r.wins,
        r.losses,
        r.decided_games,
        r.p,
        (r.p + r.z * r.z / (2 * r.n)) / (1 + r.z * r.z / r.n) as centre,
        r.z / (1 + r.z * r.z / r.n)
            * sqrt(r.p * (1 - r.p) / r.n + r.z * r.z / (4 * r.n * r.n)) as half_width
    from rated as r
)

select
    b.archetype_key || ' vs ' || b.opponent_archetype_key
        || case when b.went_first then ' going first' else ' going second' end
        as turn_order_key,
    b.archetype_key,
    mine.archetype_name,
    b.opponent_archetype_key,
    case
        when b.opponent_archetype_key = 'all' then 'All opponents'
        else theirs.archetype_name
    end as opponent_archetype_name,
    b.opponent_archetype_key = 'all' as is_all_opponents,
    b.went_first,
    b.games,
    b.wins,
    b.losses,
    b.decided_games,
    b.p as win_rate,
    -- Clamped, because Wilson is inside the unit interval by construction and
    -- a double is not: an interval that printed a hair over 1 would fail the
    -- test below for a reason that is about arithmetic and not about the data. Guarded on `decided_games` as well, because `greatest` and `least` skip
    -- nulls rather than propagating them, so a side with nothing decided
    -- would otherwise come out as the whole unit interval instead of as the
    -- absence of one.
    case when b.decided_games > 0 then greatest(0.0, b.centre - b.half_width) end as ci_low,
    case when b.decided_games > 0 then least(1.0, b.centre + b.half_width) end as ci_high,
    b.games >= {{ var('min_games') }} as min_games_met
from bounded as b
left join {{ ref('dim_archetype') }} as mine on b.archetype_key = mine.archetype_key
left join {{ ref('dim_archetype') }} as theirs on b.opponent_archetype_key = theirs.archetype_key
