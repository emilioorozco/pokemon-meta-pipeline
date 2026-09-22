-- Every stage run the pipeline has recorded, as a view over the Parquet files.
--
-- Guarded against an empty lake. `read_parquet` over a glob that matches
-- nothing is an error in DuckDB, not an empty result, so a fresh clone that has
-- never run a stage would fail `dbt run` here and take every model downstream
-- with it. The glob is probed first with DuckDB's own `glob()` table function,
-- which answers zero rather than raising, and the model falls back to a typed
-- empty relation. Same shape either way, so `mart_pipeline_health` and the
-- tests below build on an empty lake exactly as they do on a full one.
--
-- The probe runs at execution time, not at parse time, because `run_query`
-- needs a connection; `execute` is false during parsing and the fallback
-- branch is what gets compiled then.
{%- set lake = env_var('PIPELINE_DATA_DIR', 'data') ~ '/lake/run_metrics/*.parquet' -%}
{%- set files = 0 -%}
{%- if execute -%}
    {%- set probed = run_query("select count(*) as n from glob('" ~ lake ~ "')") -%}
    {%- set files = probed.columns[0].values()[0] -%}
{%- endif %}

{% if files > 0 %}

select
    run_id,
    stage,
    started_at,
    finished_at,
    duration_s,
    rows_in,
    rows_out,
    rows_quarantined,
    status,
    error,
    extra_json,
    git_commit,
    hostname
from {{ source('ops', 'run_metrics') }}

{% else %}

-- No stage has run against this data directory yet.
select
    cast(null as varchar) as run_id,
    cast(null as varchar) as stage,
    cast(null as timestamp with time zone) as started_at,
    cast(null as timestamp with time zone) as finished_at,
    cast(null as double) as duration_s,
    cast(null as bigint) as rows_in,
    cast(null as bigint) as rows_out,
    cast(null as bigint) as rows_quarantined,
    cast(null as varchar) as status,
    cast(null as varchar) as error,
    cast(null as varchar) as extra_json,
    cast(null as varchar) as git_commit,
    cast(null as varchar) as hostname
where false

{% endif %}
