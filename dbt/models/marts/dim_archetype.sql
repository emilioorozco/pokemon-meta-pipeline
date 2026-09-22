-- One row per archetype, keyed by the shared archetype row's id when the game
-- carries one and by the canonical name otherwise (see stg_game_sides for the
-- `name:` prefix and why the two key spaces cannot collide).
--
-- `archetype_name` is taken with `max()` rather than resolved again: silver
-- has already collapsed every label for an id onto one canonical name through
-- the alias map it builds from bronze, so the group is single valued and
-- `max()` only picks the one value out. `aliases` is the other half of that
-- story, every raw label the id has ever arrived under, kept so a rename is
-- visible here instead of erased.
select
    archetype_key,
    max(archetype_name) as archetype_name,
    archetype_key like 'name:%' as is_name_keyed,
    array_to_string(
        list_sort(list_distinct(list(archetype_name_raw) filter (where archetype_name_raw is not null))),
        ' | '
    ) as aliases,
    min(play_date) as first_seen,
    max(play_date) as last_seen,
    count(*) as games_played
from {{ ref('stg_game_sides') }}
where archetype_key is not null
group by archetype_key
