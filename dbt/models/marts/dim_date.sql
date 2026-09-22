-- One row per play date that has at least one game. Not a full calendar: a
-- gap-free date spine matters when a report has to show empty days, and
-- nothing here does yet, so the dimension is the dates the facts actually use.
--
-- The week columns are International Organization for Standardization (ISO)
-- week numbering, so a week always starts on a Monday and the last days of
-- December can belong to week 1 of the next `iso_year`. That is why
-- `iso_year` exists next to `year`: grouping by calendar year and ISO week
-- splits one week in two most Januaries.
select
    play_date as date_key,
    play_date,
    cast(extract(isoyear from play_date) as integer) as iso_year,
    cast(extract(week from play_date) as integer) as iso_week,
    cast(date_trunc('week', play_date) as date) as week_start,
    cast(extract(year from play_date) as integer) as year,
    cast(extract(month from play_date) as integer) as month,
    strftime(play_date, '%B') as month_name,
    cast(extract(isodow from play_date) as integer) as iso_day_of_week,
    strftime(play_date, '%A') as day_name
from (select distinct play_date from {{ ref('stg_games') }}) as dates
