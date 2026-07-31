begin;

-- OBS-001B reports only request outcomes that were explicitly admitted to the
-- recorded-rate denominator.  Time bounds use the Edge server's started_at,
-- not client-controlled event timestamps.  p_request_id provides exact
-- correlation for trusted operator verification without exposing raw events.
create function public.feedback_report_ingestion_summary(
    p_from timestamptz default null,
    p_to timestamptz default null,
    p_request_id uuid default null
)
returns table (
    recorded_request_count bigint,
    first_started_at timestamptz,
    last_finished_at timestamptz,
    success_request_count bigint,
    client_rejection_count bigint,
    service_failure_count bigint,
    accepted_event_count bigint,
    duplicate_event_count bigint,
    known_event_result_count bigint,
    request_failure_rate numeric,
    duplicate_event_rate numeric,
    recorded_rate_available boolean,
    end_to_end_coverage_available boolean
)
language sql
stable
security invoker
set search_path = ''
as $function$
with filtered as (
    select outcomes.*
    from public.feedback_ingest_request_outcomes as outcomes
    where outcomes.included_in_rate
      and (p_request_id is null or outcomes.request_id = p_request_id)
      and (p_from is null or outcomes.started_at >= p_from)
      and (p_to is null or outcomes.started_at < p_to)
      and (p_from is null or p_to is null or p_from < p_to)
),
metrics as (
    select
        count(*) as recorded_request_count,
        min(started_at) as first_started_at,
        max(finished_at) as last_finished_at,
        count(*) filter (
            where outcome_class = 'success'
        ) as success_request_count,
        count(*) filter (
            where outcome_class = 'client_rejection'
        ) as client_rejection_count,
        count(*) filter (
            where outcome_class = 'service_failure'
        ) as service_failure_count,
        coalesce(sum(accepted_event_count), 0) as accepted_event_count,
        coalesce(sum(duplicate_event_count), 0) as duplicate_event_count
    from filtered
)
select
    metrics.recorded_request_count,
    metrics.first_started_at,
    metrics.last_finished_at,
    metrics.success_request_count,
    metrics.client_rejection_count,
    metrics.service_failure_count,
    metrics.accepted_event_count,
    metrics.duplicate_event_count,
    metrics.accepted_event_count + metrics.duplicate_event_count
        as known_event_result_count,
    round(
        (
            metrics.client_rejection_count
            + metrics.service_failure_count
        )::numeric
        / nullif(metrics.recorded_request_count, 0),
        4
    ) as request_failure_rate,
    round(
        metrics.duplicate_event_count::numeric
        / nullif(
            metrics.accepted_event_count + metrics.duplicate_event_count,
            0
        ),
        4
    ) as duplicate_event_rate,
    metrics.recorded_request_count > 0 as recorded_rate_available,
    false as end_to_end_coverage_available
from metrics;
$function$;

revoke all on function public.feedback_report_ingestion_summary(
    timestamptz,
    timestamptz,
    uuid
)
from public, anon, authenticated, service_role;

grant execute on function public.feedback_report_ingestion_summary(
    timestamptz,
    timestamptz,
    uuid
)
to service_role;

comment on function public.feedback_report_ingestion_summary(
    timestamptz,
    timestamptz,
    uuid
) is 'Private OBS-001B aggregate with optional exact request correlation over persisted authenticated POST outcomes.';

commit;
