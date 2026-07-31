begin;

-- STATS-002B defines the logical event identity used by the server when an
-- event_id is replayed.  The client deliberately treats occurred_at and
-- sequence as recording metadata, while trace/request/receive fields describe
-- transport rather than the event itself.  JSONB equality makes object-key
-- order irrelevant without relying on Python/JavaScript hash canonicalization.
create function public.feedback_logical_event_v1(
    p_schema_version text,
    p_event_id text,
    p_event_type text,
    p_session_id text,
    p_question_id text,
    p_question_version text,
    p_payload jsonb
)
returns jsonb
language sql
immutable
strict
set search_path = ''
as $function$
select pg_catalog.jsonb_build_object(
    'schema_version', p_schema_version,
    'event_id', p_event_id,
    'event_type', p_event_type,
    'session_id', p_session_id,
    'question_id', p_question_id,
    'question_version', p_question_version,
    'payload', p_payload
);
$function$;

revoke all on function public.feedback_logical_event_v1(
    text,
    text,
    text,
    text,
    text,
    text,
    jsonb
)
from public, anon, authenticated, service_role;

grant execute on function public.feedback_logical_event_v1(
    text,
    text,
    text,
    text,
    text,
    text,
    jsonb
)
to service_role;

comment on function public.feedback_logical_event_v1(
    text,
    text,
    text,
    text,
    text,
    text,
    jsonb
) is 'Canonical JSONB logical-event projection for exact event-id replay comparison; excludes occurrence, sequence, trace, request, and receive metadata.';

-- One row records one conflicting event-id observation.  It intentionally
-- stores neither an incoming payload nor a content hash: accepted content
-- remains in feedback_events and exact comparison stays inside the ingest RPC.
create table public.feedback_event_conflicts (
    request_id uuid not null,
    event_id text not null,
    first_ingest_request_id uuid not null,
    comparison_revision text not null,
    detected_at timestamptz not null default pg_catalog.statement_timestamp(),

    constraint feedback_event_conflicts_pkey
        primary key (request_id, event_id),
    constraint feedback_event_conflicts_event_id_fkey
        foreign key (event_id)
        references public.feedback_events (event_id),
    constraint feedback_event_conflicts_event_id_check
        check (
            event_id = btrim(event_id)
            and length(event_id) between 1 and 200
            and event_id !~ E'[\r\n]'
        ),
    constraint feedback_event_conflicts_revision_check
        check (comparison_revision = 'logical_event_v1')
);

comment on table public.feedback_event_conflicts is
    'Private append-only audit of same event_id with different logical content; contains no event payload or content hash.';
comment on column public.feedback_event_conflicts.request_id is
    'Server-generated ingestion request UUID that observed the conflict; it is not foreign-keyed to fail-open request outcomes.';
comment on column public.feedback_event_conflicts.first_ingest_request_id is
    'ingest_request_id of the immutable first-write-wins feedback_events row.';

create index feedback_event_conflicts_detected_at_idx
    on public.feedback_event_conflicts (detected_at desc);
create index feedback_event_conflicts_event_id_idx
    on public.feedback_event_conflicts (event_id, detected_at desc);

alter table public.feedback_event_conflicts enable row level security;
alter table public.feedback_event_conflicts force row level security;

revoke all on table public.feedback_event_conflicts
    from public, anon, authenticated, service_role;
grant select on table public.feedback_event_conflicts to service_role;

create function public.reject_feedback_event_conflict_mutation()
returns trigger
language plpgsql
set search_path = ''
as $function$
begin
    raise exception using
        errcode = '55000',
        message = 'feedback_event_conflicts is append-only';
end;
$function$;

revoke all on function public.reject_feedback_event_conflict_mutation()
    from public, anon, authenticated, service_role;

create trigger feedback_event_conflicts_append_only
before update or delete or truncate on public.feedback_event_conflicts
for each statement execute function
    public.reject_feedback_event_conflict_mutation();

-- The RPC is the concurrency boundary for event-id classification and event
-- insertion.  It acquires every transaction-scoped advisory lock in numeric
-- key order before reading existing rows, so overlapping batches cannot take
-- the same lock set in opposite orders.  The later lockdown migration makes
-- this the only service-role write path; the RETURNING check fails closed if a
-- legacy/direct writer races during the deployment transition.
create function public.feedback_ingest_events(
    p_request_id uuid,
    p_trace_id text,
    p_trace_created_at timestamptz,
    p_events jsonb
)
returns table (
    requested_event_count integer,
    new_event_count integer,
    accepted_event_count integer,
    duplicate_event_count integer,
    conflicting_event_count integer,
    rejected_event_count integer,
    committed boolean
)
language plpgsql
volatile
security definer
set search_path = ''
as $function$
declare
    v_requested integer;
    v_new integer;
    v_duplicate integer;
    v_conflicting integer;
    v_inserted integer;
    v_lock_key bigint;
    v_distinct_event_ids integer;
    v_session_count integer;
begin
    if p_request_id is null then
        raise exception using
            errcode = '22023',
            message = 'p_request_id must not be null';
    end if;
    if (
        p_trace_id is null
        or p_trace_id <> pg_catalog.btrim(p_trace_id)
        or pg_catalog.length(p_trace_id) not between 1 and 200
        or p_trace_id ~ E'[\r\n]'
    ) then
        raise exception using
            errcode = '22023',
            message = 'p_trace_id is invalid';
    end if;
    if p_trace_created_at is null then
        raise exception using
            errcode = '22023',
            message = 'p_trace_created_at must not be null';
    end if;
    if pg_catalog.jsonb_typeof(p_events) is distinct from 'array' then
        raise exception using
            errcode = '22023',
            message = 'p_events must be a JSON array';
    end if;

    v_requested := pg_catalog.jsonb_array_length(p_events);
    if v_requested not between 1 and 500 then
        raise exception using
            errcode = '22023',
            message = 'p_events must contain between 1 and 500 events';
    end if;

    if exists (
        select 1
        from pg_catalog.jsonb_array_elements(p_events) as items(event)
        where pg_catalog.jsonb_typeof(items.event) <> 'object'
    ) then
        raise exception using
            errcode = '22023',
            message = 'every p_events item must be an object';
    end if;

    if exists (
        select 1
        from pg_catalog.jsonb_array_elements(p_events) as items(event)
        where not items.event ?& array[
                'schema_version',
                'event_id',
                'event_type',
                'occurred_at',
                'session_id',
                'question_id',
                'question_version',
                'payload',
                'sequence'
            ]::text[]
           or (
                select pg_catalog.count(*)
                from pg_catalog.jsonb_object_keys(items.event) as keys(key)
            ) <> 9
    ) then
        raise exception using
            errcode = '22023',
            message = 'p_events items must use the exact event field set';
    end if;

    if exists (
        select 1
        from pg_catalog.jsonb_array_elements(p_events) as items(event)
        where pg_catalog.jsonb_typeof(items.event -> 'schema_version') <> 'string'
           or pg_catalog.jsonb_typeof(items.event -> 'event_id') <> 'string'
           or pg_catalog.jsonb_typeof(items.event -> 'event_type') <> 'string'
           or pg_catalog.jsonb_typeof(items.event -> 'occurred_at') <> 'string'
           or pg_catalog.jsonb_typeof(items.event -> 'session_id') <> 'string'
           or pg_catalog.jsonb_typeof(items.event -> 'question_id') <> 'string'
           or pg_catalog.jsonb_typeof(items.event -> 'question_version') <> 'string'
           or pg_catalog.jsonb_typeof(items.event -> 'payload') <> 'object'
           or pg_catalog.jsonb_typeof(items.event -> 'sequence') <> 'number'
    ) then
        raise exception using
            errcode = '22023',
            message = 'p_events contains an invalid field type';
    end if;

    with input as (
        select events.*
        from pg_catalog.jsonb_to_recordset(p_events) as events(
            schema_version text,
            event_id text,
            event_type text,
            occurred_at timestamptz,
            session_id text,
            question_id text,
            question_version text,
            payload jsonb,
            sequence numeric
        )
    )
    select
        pg_catalog.count(distinct input.event_id)::integer,
        pg_catalog.count(distinct input.session_id)::integer
    into v_distinct_event_ids, v_session_count
    from input;

    if v_distinct_event_ids <> v_requested then
        raise exception using
            errcode = '22023',
            message = 'p_events contains duplicate event_id values';
    end if;
    if v_session_count <> 1 then
        raise exception using
            errcode = '22023',
            message = 'p_events must contain exactly one session_id';
    end if;

    if exists (
        with input as (
            select events.*
            from pg_catalog.jsonb_to_recordset(p_events) as events(
                schema_version text,
                event_id text,
                event_type text,
                occurred_at timestamptz,
                session_id text,
                question_id text,
                question_version text,
                payload jsonb,
                sequence numeric
            )
        )
        select 1
        from input
        where input.schema_version <> '1.0'
           or input.event_id <> pg_catalog.btrim(input.event_id)
           or pg_catalog.length(input.event_id) not between 1 and 200
           or input.event_id ~ E'[\r\n]'
           or input.event_type not in (
                'answer_submitted',
                'custom_setting_proposed',
                'custom_setting_rejected',
                'custom_run_completed',
                'custom_run_failed',
                'comment_submitted'
            )
           or input.occurred_at is null
           or input.session_id <> pg_catalog.btrim(input.session_id)
           or pg_catalog.length(input.session_id) not between 1 and 200
           or input.session_id ~ E'[\r\n]'
           or input.question_id <> pg_catalog.btrim(input.question_id)
           or pg_catalog.length(input.question_id) not between 1 and 200
           or input.question_id ~ E'[\r\n]'
           or input.question_version <> pg_catalog.btrim(input.question_version)
           or pg_catalog.length(input.question_version) not between 1 and 200
           or input.question_version ~ E'[\r\n]'
           or pg_catalog.jsonb_typeof(input.payload) <> 'object'
           or input.sequence <> pg_catalog.trunc(input.sequence)
           or input.sequence not between 1 and 2147483647
    ) then
        raise exception using
            errcode = '22023',
            message = 'p_events contains an invalid event';
    end if;

    if exists (
        with ordered as (
            select
                items.ordinality,
                (items.event ->> 'sequence')::integer as sequence,
                pg_catalog.lag((items.event ->> 'sequence')::integer)
                    over (order by items.ordinality) as previous_sequence
            from pg_catalog.jsonb_array_elements(p_events)
                with ordinality as items(event, ordinality)
        )
        select 1
        from ordered
        where ordered.previous_sequence is not null
          and ordered.sequence <= ordered.previous_sequence
    ) then
        raise exception using
            errcode = '22023',
            message = 'p_events sequence values must be strictly increasing';
    end if;

    -- Sort the actual advisory key, not event_id.  Sorting event_id alone can
    -- still produce opposite lock-key orders for partially overlapping sets.
    for v_lock_key in
        with input as (
            select events.event_id
            from pg_catalog.jsonb_to_recordset(p_events) as events(
                event_id text
            )
        ), lock_keys as (
            select distinct pg_catalog.hashtextextended(
                'architecture_iq.feedback_event.logical_event_v1:'
                    || input.event_id,
                0::bigint
            ) as lock_key
            from input
        )
        select lock_keys.lock_key
        from lock_keys
        order by lock_keys.lock_key
    loop
        perform pg_catalog.pg_advisory_xact_lock(v_lock_key);
    end loop;

    with input as (
        select events.*
        from pg_catalog.jsonb_to_recordset(p_events) as events(
            schema_version text,
            event_id text,
            event_type text,
            occurred_at timestamptz,
            session_id text,
            question_id text,
            question_version text,
            payload jsonb,
            sequence numeric
        )
    ), classified as (
        select
            stored.event_id is null as is_new,
            stored.event_id is not null
                and public.feedback_logical_event_v1(
                    stored.schema_version,
                    stored.event_id,
                    stored.event_type,
                    stored.session_id,
                    stored.question_id,
                    stored.question_version,
                    stored.payload
                ) = public.feedback_logical_event_v1(
                    input.schema_version,
                    input.event_id,
                    input.event_type,
                    input.session_id,
                    input.question_id,
                    input.question_version,
                    input.payload
                ) as is_duplicate
        from input
        left join public.feedback_events as stored
            on stored.event_id = input.event_id
    )
    select
        pg_catalog.count(*) filter (where classified.is_new)::integer,
        pg_catalog.count(*) filter (
            where not classified.is_new and classified.is_duplicate
        )::integer,
        pg_catalog.count(*) filter (
            where not classified.is_new and not classified.is_duplicate
        )::integer
    into v_new, v_duplicate, v_conflicting
    from classified;

    if v_conflicting > 0 then
        with input as (
            select events.*
            from pg_catalog.jsonb_to_recordset(p_events) as events(
                schema_version text,
                event_id text,
                event_type text,
                occurred_at timestamptz,
                session_id text,
                question_id text,
                question_version text,
                payload jsonb,
                sequence numeric
            )
        )
        insert into public.feedback_event_conflicts (
            request_id,
            event_id,
            first_ingest_request_id,
            comparison_revision,
            detected_at
        )
        select
            p_request_id,
            stored.event_id,
            stored.ingest_request_id,
            'logical_event_v1',
            pg_catalog.statement_timestamp()
        from input
        join public.feedback_events as stored
            on stored.event_id = input.event_id
        where public.feedback_logical_event_v1(
                stored.schema_version,
                stored.event_id,
                stored.event_type,
                stored.session_id,
                stored.question_id,
                stored.question_version,
                stored.payload
            ) <> public.feedback_logical_event_v1(
                input.schema_version,
                input.event_id,
                input.event_type,
                input.session_id,
                input.question_id,
                input.question_version,
                input.payload
            )
        on conflict (request_id, event_id) do nothing;

        return query select
            v_requested,
            v_new,
            0,
            v_duplicate,
            v_conflicting,
            v_requested - v_duplicate,
            false;
        return;
    end if;

    with input as (
        select events.*
        from pg_catalog.jsonb_to_recordset(p_events) as events(
            schema_version text,
            event_id text,
            event_type text,
            occurred_at timestamptz,
            session_id text,
            question_id text,
            question_version text,
            payload jsonb,
            sequence numeric
        )
    ), new_rows as (
        select input.*
        from input
        left join public.feedback_events as stored
            on stored.event_id = input.event_id
        where stored.event_id is null
        order by input.event_id
    ), inserted as (
        insert into public.feedback_events (
            event_id,
            schema_version,
            trace_id,
            trace_created_at,
            session_id,
            question_id,
            question_version,
            event_type,
            occurred_at,
            sequence,
            payload,
            ingest_request_id
        )
        select
            new_rows.event_id,
            new_rows.schema_version,
            p_trace_id,
            p_trace_created_at,
            new_rows.session_id,
            new_rows.question_id,
            new_rows.question_version,
            new_rows.event_type,
            new_rows.occurred_at,
            new_rows.sequence::integer,
            new_rows.payload,
            p_request_id
        from new_rows
        on conflict (event_id) do nothing
        returning event_id
    )
    select pg_catalog.count(*)::integer
    into v_inserted
    from inserted;

    if v_inserted <> v_new then
        raise exception using
            errcode = '40001',
            message = 'feedback event writer race detected; retry the request';
    end if;

    return query select
        v_requested,
        v_new,
        v_inserted,
        v_duplicate,
        0,
        0,
        true;
end;
$function$;

revoke all on function public.feedback_ingest_events(
    uuid,
    text,
    timestamptz,
    jsonb
)
from public, anon, authenticated, service_role;

grant execute on function public.feedback_ingest_events(
    uuid,
    text,
    timestamptz,
    jsonb
)
to service_role;

comment on function public.feedback_ingest_events(
    uuid,
    text,
    timestamptz,
    jsonb
) is 'Service-role-only atomic STATS-002B ingest: exact logical duplicate comparison, conflict audit, and all-or-none feedback-event insertion on conflict.';

-- Forward-compatible outcome revision.  Existing append-only 1.0 rows keep a
-- NULL conflict count; the upgraded Edge writes 1.1.  Validation rejection,
-- storage-unknown, and excluded 1.1 outcomes keep the count NULL because event
-- comparison did not produce a confirmed result.
alter table public.feedback_ingest_request_outcomes
    add column conflicting_event_count integer;

alter table public.feedback_ingest_request_outcomes
    drop constraint feedback_ingest_outcomes_schema_version_check,
    drop constraint feedback_ingest_outcomes_code_check,
    drop constraint feedback_ingest_outcomes_count_check,
    drop constraint feedback_ingest_outcomes_classification_check;

alter table public.feedback_ingest_request_outcomes
    add constraint feedback_ingest_outcomes_schema_version_check
        check (schema_version in ('1.0', '1.1')),
    add constraint feedback_ingest_outcomes_code_check
        check (
            outcome_code in (
                'accepted_only',
                'duplicate_only',
                'mixed_success',
                'event_id_conflict',
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
    add constraint feedback_ingest_outcomes_count_check
        check (
            (requested_event_count is null or requested_event_count >= 0)
            and (accepted_event_count is null or accepted_event_count >= 0)
            and (duplicate_event_count is null or duplicate_event_count >= 0)
            and (conflicting_event_count is null or conflicting_event_count >= 0)
            and (rejected_event_count is null or rejected_event_count >= 0)
        ),
    add constraint feedback_ingest_outcomes_revision_conflict_check
        check (
            (
                schema_version = '1.0'
                and conflicting_event_count is null
            )
            or (
                schema_version = '1.1'
                and (
                    (
                        outcome_code in (
                            'accepted_only',
                            'duplicate_only',
                            'mixed_success'
                        )
                        and conflicting_event_count = 0
                    )
                    or (
                        outcome_code = 'event_id_conflict'
                        and conflicting_event_count > 0
                    )
                    or (
                        outcome_code in (
                            'request_too_large',
                            'invalid_request',
                            'invalid_envelope',
                            'storage_unavailable',
                            'internal_error',
                            'method_not_allowed',
                            'unauthorized',
                            'service_unavailable'
                        )
                        and conflicting_event_count is null
                    )
                )
            )
        ),
    add constraint feedback_ingest_outcomes_classification_check
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
                outcome_class = 'client_rejection'
                and outcome_code = 'event_id_conflict'
                and method = 'POST'
                and authenticated
                and included_in_rate
                and http_status = 409
                and submission_kind in ('session_trace', 'single_comment')
                and requested_event_count is not null
                and requested_event_count > 0
                and accepted_event_count is not null
                and accepted_event_count = 0
                and duplicate_event_count is not null
                and conflicting_event_count is not null
                and conflicting_event_count > 0
                and rejected_event_count is not null
                and rejected_event_count > 0
                and requested_event_count
                    = duplicate_event_count + rejected_event_count
                and conflicting_event_count <= rejected_event_count
                and storage_state = 'confirmed'
                and not retryable
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
        );

comment on column public.feedback_ingest_request_outcomes.conflicting_event_count is
    'Confirmed same-event-id/different-logical-content count for schema 1.1; NULL for legacy rows and requests without a confirmed comparison result.';

commit;
