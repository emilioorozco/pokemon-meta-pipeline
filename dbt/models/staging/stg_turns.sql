-- One row per turn segment. Renames and casts only.
--
-- Two attributions live side by side here, and reading the wrong one is the
-- easy mistake. The plain `n_*` counters count every line printed in the
-- segment and hang them all on the seat whose turn it was. The `n_*_self` and
-- `n_*_opp` pairs count four of those kinds again by the seat the log credits
-- each line to, which is not the same seat: a Pokemon can be knocked out on
-- its own owner's turn, and the knockout and the prizes for it then belong to
-- the other side of the table. `int_game_side_pace` reads the pairs, because
-- the application's pace definitions are written in terms of credit.
select
    game_id,
    cast(play_date as date) as play_date,
    cast(turn_number as integer) as turn_number,
    cast(seat as integer) as seat,
    cast(n_entries as integer) as n_entries,
    cast(n_draw as integer) as n_draw,
    cast(n_attach as integer) as n_attach,
    cast(n_attack as integer) as n_attack,
    cast(n_play_pokemon as integer) as n_play_pokemon,
    cast(n_play_trainer as integer) as n_play_trainer,
    cast(n_evolve as integer) as n_evolve,
    cast(n_retreat as integer) as n_retreat,
    cast(n_knockout as integer) as n_knockout,
    cast(n_prize_taken as integer) as n_prize_taken,
    cast(n_attack_self as integer) as n_attack_self,
    cast(n_attack_opp as integer) as n_attack_opp,
    cast(n_energy_attach_self as integer) as n_energy_attach_self,
    cast(n_energy_attach_opp as integer) as n_energy_attach_opp,
    cast(n_prize_self as integer) as n_prize_self,
    cast(n_prize_opp as integer) as n_prize_opp,
    cast(n_knockout_self as integer) as n_knockout_self,
    cast(n_knockout_opp as integer) as n_knockout_opp,
    coalesce(concession, false) as concession
from {{ source('silver', 'turns') }}
