begin;

-- OBS-001 stores one sanitized result per observed ingestion request.  This is
-- deliberately separate from feedback_events so request telemetry cannot
-- change accepted-event history or quiz/report aggregates.
create table public.feedback_ingest_request_outcomes (
    request_id uuid primary key,
    schema_version text not null,
    started_at timestamptz not null,
    finished_at timestamptz not null,
    duration_ms integer not null,
    method text not null,
    authenticated boolean not null,
    included_in_rate boolean not null,
    outcome_class text not null,
    outcome_code text not null,
    http_status integer not null,
    submission_kind text not null,
    requested_event_count integer,
    accepted_event_count integer,
    duplicate_event_count integer,
    rejected_event_count integer,
    storage_state text not null,
    retryable boolean not null,
    observer_revision text not null,
    recorded_at timestamptz not null default now(),

    constraint feedback_ingest_outcomes_schema_version_check
        check (schema_version = '1.0'),
    constraint feedback_ingest_outcomes_time_check
        check (finished_at >= started_at and duration_ms >= 0),
    constraint feedback_ingest_outcomes_method_check
        check (
            method = btrim(method)
            and method ~ '^[A-Z]{1,16}$'
        ),
    constraint feedback_ingest_outcomes_class_check
        check (
            outcome_class in (
                'success',
                'client_rejection',
                'service_failure',
                'excluded'
            )
        ),
    constraint feedback_ingest_outcomes_code_check
        check (
            outcome_code in (
                'accepted_only',
                'duplicate_only',
                'mixed_success',
                'request_too_large',
                'invalid_request',
                'invalid_envelope',
                'storage_unavailable',
                'internal_error',
                'method_not_allowed',
                'unauthorized',
                'service_unavailable'
            )
        ),
    constraint feedback_ingest_outcomes_submission_kind_check
        check (
            submission_kind in (
                'session_trace',
                'single_comment',
                'unknown'
            )
        ),
    constraint feedback_ingest_outcomes_count_check
        check (
            (requested_event_count is null or requested_event_count >= 0)
            and (accepted_event_count is null or accepted_event_count >= 0)
            and (duplicate_event_count is null or duplicate_event_count >= 0)
            and (rejected_event_count is null or rejected_event_count >= 0)
        ),
    constraint feedback_ingest_outcomes_storage_state_check
        check (
            storage_state in (
                'confirmed',
                'not_attempted',
                'not_committed',
                'unknown'
            )
        ),
    constraint feedback_ingest_outcomes_observer_revision_check
        check (
            observer_revision = btrim(observer_revision)
            and length(observer_revision) between 1 and 200
            and observer_revision !~ E'[\r\n]'
        ),
    constraint feedback_ingest_outcomes_classification_check
        check (
            (
                outcome_class = 'success'
                and outcome_code in (
                    'accepted_only',
                    'duplicate_only',
                    'mixed_success'
                )
                and method = 'POST'
                and authenticated
                and included_in_rate
                and http_status = 200
                and submission_kind in ('session_trace', 'single_comment')
                and requested_event_count is not null
                and requested_event_count > 0
                and accepted_event_count is not null
                and duplicate_event_count is not null
                and rejected_event_count is not null
                and rejected_event_count = 0
                and requested_event_count
                    = accepted_event_count + duplicate_event_count
                and storage_state = 'confirmed'
                and not retryable
                and (
                    (
                        outcome_code = 'accepted_only'
                        and accepted_event_count > 0
                        and duplicate_event_count = 0
                    )
                    or (
                        outcome_code = 'duplicate_only'
                        and accepted_event_count = 0
                        and duplicate_event_count > 0
                    )
                    or (
                        outcome_code = 'mixed_success'
                        and accepted_event_count > 0
                        and duplicate_event_count > 0
                    )
                )
            )
            or (
                outcome_class = 'client_rejection'
                and outcome_code in (
                    'request_too_large',
                    'invalid_request',
                    'invalid_envelope'
                )
                and method = 'POST'
                and authenticated
                and included_in_rate
                and (
                    (
                        outcome_code = 'request_too_large'
                        and http_status = 413
                    )
                    or (
                        outcome_code in ('invalid_request', 'invalid_envelope')
                        and http_status = 400
                    )
                )
                and submission_kind = 'unknown'
                and accepted_event_count is not null
                and accepted_event_count = 0
                and duplicate_event_count is not null
                and duplicate_event_count = 0
                and rejected_event_count is not null
                and storage_state = 'not_attempted'
                and not retryable
                and (
                    (
                        outcome_code in ('request_too_large', 'invalid_request')
                        and requested_event_count is null
                        and rejected_event_count = 0
                    )
                    or (
                        outcome_code = 'invalid_envelope'
                        and (
                            (
                                requested_event_count is null
                                and rejected_event_count = 0
                            )
                            or (
                                requested_event_count is not null
                                and requested_event_count = rejected_event_count
                            )
                        )
                    )
                )
            )
            or (
                outcome_class = 'service_failure'
                and outcome_code in ('storage_unavailable', 'internal_error')
                and method = 'POST'
                and authenticated
                and included_in_rate
                and (
                    (outcome_code = 'storage_unavailable' and http_status = 502)
                    or (outcome_code = 'internal_error' and http_status = 500)
                )
                and storage_state in (
                    'not_attempted',
                    'not_committed',
                    'unknown'
                )
                and retryable
                and (
                    (
                        outcome_code = 'storage_unavailable'
                        and submission_kind in (
                            'session_trace',
                            'single_comment'
                        )
                        and requested_event_count is not null
                        and requested_event_count > 0
                        and accepted_event_count is null
                        and duplicate_event_count is null
                        and rejected_event_count is null
                    )
                    or (
                        outcome_code = 'internal_error'
                        and submission_kind = 'unknown'
                        and requested_event_count is null
                        and accepted_event_count is null
                        and duplicate_event_count is null
                        and rejected_event_count is null
                    )
                )
            )
            or (
                outcome_class = 'excluded'
                and outcome_code in (
                    'method_not_allowed',
                    'unauthorized',
                    'service_unavailable'
                )
                and not authenticated
                and not included_in_rate
                and submission_kind = 'unknown'
                and requested_event_count is null
                and accepted_event_count is null
                and duplicate_event_count is null
                and rejected_event_count is null
                and storage_state = 'not_attempted'
                and (
                    (
                        outcome_code = 'method_not_allowed'
                        and method <> 'POST'
                        and http_status = 405
                        and not retryable
                    )
                    or (
                        outcome_code = 'unauthorized'
                        and method = 'POST'
                        and http_status = 401
                        and not retryable
                    )
                    or (
                        outcome_code = 'service_unavailable'
                        and method = 'POST'
                        and http_status = 503
                        and retryable
                    )
                )
            )
        )
);

create index feedback_ingest_outcomes_started_at_idx
    on public.feedback_ingest_request_outcomes (started_at desc);
create index feedback_ingest_outcomes_class_time_idx
    on public.feedback_ingest_request_outcomes (
        outcome_class,
        started_at desc
    );
create index feedback_ingest_outcomes_code_time_idx
    on public.feedback_ingest_request_outcomes (
        outcome_code,
        started_at desc
    );
create index feedback_ingest_outcomes_included_time_idx
    on public.feedback_ingest_request_outcomes (started_at desc)
    where included_in_rate;

alter table public.feedback_ingest_request_outcomes enable row level security;
alter table public.feedback_ingest_request_outcomes force row level security;

revoke all on table public.feedback_ingest_request_outcomes
    from public, anon, authenticated, service_role;
grant select, insert on table public.feedback_ingest_request_outcomes
    to service_role;

create function public.reject_feedback_ingest_outcome_mutation()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    raise exception using
        errcode = '55000',
        message = 'feedback_ingest_request_outcomes is append-only';
end;
$$;

revoke all on function public.reject_feedback_ingest_outcome_mutation()
    from public, anon, authenticated;

create trigger feedback_ingest_request_outcomes_append_only
before update or delete or truncate
on public.feedback_ingest_request_outcomes
for each statement execute function
    public.reject_feedback_ingest_outcome_mutation();

commit;
