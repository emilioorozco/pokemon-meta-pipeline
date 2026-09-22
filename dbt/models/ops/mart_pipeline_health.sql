-- One row per stage: how its last run went, and how much it has been rejecting.
--
-- The question this answers is the one asked in front of a failed morning DAG
-- run: which stage broke, how long it normally takes, and has it been throwing
-- rows away for a while. So the row carries both a point (the last run) and a
-- trend (the last ten), because a stage whose last run was green after nine
-- red ones is not a healthy stage.
--
-- The quarantine rate is over the window rather than per run: a run that read
-- three objects and rejected one is 33%, and averaging that against a run of
-- ten thousand would let a tiny run shout down a large one. Summing both sides
-- first weights every row equally, which is what a rate is supposed to mean.
-- Stages that quarantine nothing by construction (silver, gold, train) report
-- a rate of zero rather than null, so the column can be charted across stages.
{% set window = 10 %}
with ranked as (
    select
        *,
        -- `run_id` breaks a tie: two stages of one DAG run can start in the
        -- same microsecond, and a window function with a non-deterministic
        -- order would pick a different "last run" on every query.
        row_number() over (partition by stage order by started_at desc, run_id desc) as recency
    from {{ ref('run_metrics') }}
),

latest as (
    select * from ranked where recency = 1
),

recent as (
    select
        stage,
        count(*) as runs_considered,
        count(*) filter (where status = 'failed') as failed_runs,
        sum(coalesce(rows_in, 0)) as rows_in_window,
        sum(coalesce(rows_quarantined, 0)) as rows_quarantined_window,
        avg(duration_s) as mean_duration_s
    from ranked
    where recency <= {{ window }}
    group by stage
)

select
    l.stage,
    l.run_id as last_run_id,
    l.status as last_status,
    l.started_at as last_started_at,
    l.duration_s as last_duration_s,
    l.rows_in as last_rows_in,
    l.rows_out as last_rows_out,
    l.rows_quarantined as last_rows_quarantined,
    l.error as last_error,
    l.git_commit as last_git_commit,
    r.runs_considered,
    r.failed_runs,
    r.mean_duration_s,
    r.rows_in_window,
    r.rows_quarantined_window,
    coalesce(
        r.rows_quarantined_window / nullif(cast(r.rows_in_window as double), 0), 0.0
    ) as quarantine_rate,
    coalesce(
        r.rows_quarantined_window / nullif(cast(r.rows_in_window as double), 0), 0.0
    ) >= {{ var('quarantine_rate_alert') }} as quarantine_rate_over_threshold
from latest as l
inner join recent as r on l.stage = r.stage
