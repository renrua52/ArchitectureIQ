begin;

-- STATS-002B extends the recorded ingestion subset with an explicit split
-- between verified idempotent retries, legacy/unclassified duplicate IDs, and
-- rejected event-ID conflicts.  The RPC name and parameters stay stable so
-- the protected report endpoint does not gain another query surface.
drop function public.feedback_report_ingestion_summary(
    timestamptz,
    timestamptz,
    uuid
);

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
    event_id_conflict_request_count bigint,
    accepted_event_count bigint,
    duplicate_event_count bigint,
    idempotent_duplicate_event_count bigint,
    unclassified_duplicate_event_count bigint,
    conflicting_event_count bigint,
    conflict_audit_event_count bigint,
    event_id_reuse_count bigint,
    classified_event_count bigint,
    known_event_result_count bigint,
    request_failure_rate numeric,
    duplicate_event_rate numeric,
    event_id_reuse_rate numeric,
    classified_conflicting_event_rate numeric,
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
        count(*) filter (
            where outcome_code = 'event_id_conflict'
        ) as event_id_conflict_request_count,
        coalesce(sum(accepted_event_count), 0) as accepted_event_count,
        coalesce(sum(duplicate_event_count), 0) as duplicate_event_count,
        coalesce(
            sum(duplicate_event_count) filter (
                where conflicting_event_count is not null
            ),
            0
        ) as idempotent_duplicate_event_count,
        coalesce(sum(conflicting_event_count), 0) as conflicting_event_count,
        coalesce(
            sum(requested_event_count) filter (
                where outcome_code in (
                    'accepted_only',
                    'duplicate_only',
                    'mixed_success',
                    'event_id_conflict'
                )
            ),
            0
        ) as classified_event_count
    from filtered
),
derived as (
    select
        metrics.*,
        metrics.duplicate_event_count
            - metrics.idempotent_duplicate_event_count
            as unclassified_duplicate_event_count,
        metrics.duplicate_event_count
            + metrics.conflicting_event_count
            as event_id_reuse_count,
        metrics.accepted_event_count
            + metrics.duplicate_event_count
            as known_event_result_count
    from metrics
), conflict_audit_metrics as (
    select count(*) as conflict_audit_event_count
    from public.feedback_event_conflicts as conflicts
    join filtered
        on filtered.request_id = conflicts.request_id
)
select
    derived.recorded_request_count,
    derived.first_started_at,
    derived.last_finished_at,
    derived.success_request_count,
    derived.client_rejection_count,
    derived.service_failure_count,
    derived.event_id_conflict_request_count,
    derived.accepted_event_count,
    derived.duplicate_event_count,
    derived.idempotent_duplicate_event_count,
    derived.unclassified_duplicate_event_count,
    derived.conflicting_event_count,
    conflict_audit_metrics.conflict_audit_event_count,
    derived.event_id_reuse_count,
    derived.classified_event_count,
    derived.known_event_result_count,
    round(
        (
            derived.client_rejection_count
            + derived.service_failure_count
        )::numeric
        / nullif(derived.recorded_request_count, 0),
        4
    ) as request_failure_rate,
    round(
        derived.duplicate_event_count::numeric
        / nullif(derived.known_event_result_count, 0),
        4
    ) as duplicate_event_rate,
    round(
        derived.event_id_reuse_count::numeric
        / nullif(derived.classified_event_count, 0),
        4
    ) as event_id_reuse_rate,
    round(
        derived.conflicting_event_count::numeric
        / nullif(
            derived.idempotent_duplicate_event_count
                + derived.conflicting_event_count,
            0
        ),
        4
    ) as classified_conflicting_event_rate,
    derived.recorded_request_count > 0 as recorded_rate_available,
    false as end_to_end_coverage_available
from derived
cross join conflict_audit_metrics;
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
) is 'Private recorded-subset ingestion aggregate with event-ID conflict classification, sidecar audit verification, and optional exact request correlation.';

commit;
