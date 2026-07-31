begin;

-- Add structured presentation and post-reveal reaction facts without changing
-- the event wire schema. Existing rows and the six original event types remain
-- valid.
alter table public.feedback_events
    drop constraint feedback_events_event_type_check,
    add constraint feedback_events_event_type_check
        check (
            event_type in (
                'answer_submitted',
                'custom_setting_proposed',
                'custom_setting_rejected',
                'custom_run_completed',
                'custom_run_failed',
                'comment_submitted',
                'question_presented',
                'question_reaction_submitted'
            )
        ),
    add constraint feedback_events_question_reaction_payload_check
        check (
            event_type <> 'question_reaction_submitted'
            or coalesce(
                (
                    payload ?& array[
                        'reaction',
                        'value',
                        'timing',
                        'attempt_id'
                    ]::text[]
                    and pg_catalog.jsonb_typeof(payload -> 'reaction') = 'string'
                    and payload ->> 'reaction' = 'surprise'
                    and pg_catalog.jsonb_typeof(payload -> 'value') = 'boolean'
                    and pg_catalog.jsonb_typeof(payload -> 'timing') = 'string'
                    and payload ->> 'timing' = 'after_reveal'
                    and pg_catalog.jsonb_typeof(payload -> 'attempt_id') = 'string'
                    and payload ->> 'attempt_id'
                        = pg_catalog.btrim(payload ->> 'attempt_id')
                    and pg_catalog.length(payload ->> 'attempt_id')
                        between 1 and 200
                    and payload ->> 'attempt_id' !~ E'[\r\n]'
                    and (
                        not (payload ? 'release_id')
                        or coalesce(
                            (
                                pg_catalog.jsonb_typeof(
                                    payload -> 'release_id'
                                ) = 'string'
                                and payload ->> 'release_id'
                                    = pg_catalog.btrim(
                                        payload ->> 'release_id'
                                    )
                                and pg_catalog.length(
                                    payload ->> 'release_id'
                                ) between 1 and 200
                                and payload ->> 'release_id' !~ E'[\r\n]'
                            ),
                            false
                        )
                    )
                ),
                false
            )
        ),
    add constraint feedback_events_question_presented_payload_check
        check (
            event_type <> 'question_presented'
            or coalesce(
                (
                    payload ?& array[
                        'attempt_id',
                        'release_id',
                        'decision_id',
                        'policy_version',
                        'mode',
                        'propensity',
                        'source',
                        'position'
                    ]::text[]
                    and pg_catalog.jsonb_typeof(payload -> 'attempt_id') = 'string'
                    and payload ->> 'attempt_id'
                        = pg_catalog.btrim(payload ->> 'attempt_id')
                    and pg_catalog.length(payload ->> 'attempt_id') between 1 and 200
                    and payload ->> 'attempt_id' !~ E'[\r\n]'
                    and pg_catalog.jsonb_typeof(payload -> 'release_id') = 'string'
                    and payload ->> 'release_id'
                        = pg_catalog.btrim(payload ->> 'release_id')
                    and pg_catalog.length(payload ->> 'release_id') between 1 and 200
                    and payload ->> 'release_id' !~ E'[\r\n]'
                    and pg_catalog.jsonb_typeof(payload -> 'decision_id') = 'string'
                    and payload ->> 'decision_id'
                        = pg_catalog.btrim(payload ->> 'decision_id')
                    and pg_catalog.length(payload ->> 'decision_id') between 1 and 200
                    and payload ->> 'decision_id' !~ E'[\r\n]'
                    and pg_catalog.jsonb_typeof(payload -> 'policy_version') = 'string'
                    and payload ->> 'policy_version'
                        = pg_catalog.btrim(payload ->> 'policy_version')
                    and pg_catalog.length(payload ->> 'policy_version') between 1 and 200
                    and payload ->> 'policy_version' !~ E'[\r\n]'
                    and payload ->> 'mode' in (
                        'exploit', 'explore', 'fallback', 'manual'
                    )
                    and pg_catalog.jsonb_typeof(payload -> 'propensity') = 'number'
                    and (payload ->> 'propensity')::numeric > 0
                    and (payload ->> 'propensity')::numeric <= 1
                    and payload ->> 'source' in (
                        'initial', 'next', 'random', 'picker'
                    )
                    and pg_catalog.jsonb_typeof(payload -> 'position') = 'number'
                    and (payload ->> 'position')::numeric
                        = pg_catalog.trunc((payload ->> 'position')::numeric)
                    and (payload ->> 'position')::numeric
                        between 1 and 9007199254740991
                ),
                false
            )
        );

-- STATS-002B validates the event type before reaching the table. Recreate the
-- same RPC contract with only the new enum added; advisory locking, logical
-- idempotency, conflict auditing, and all-or-none insertion remain unchanged.
create or replace function public.feedback_ingest_events(
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
                'comment_submitted',
                'question_presented',
                'question_reaction_submitted'
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
) is 'Service-role-only atomic STATS-002B ingest with question reactions, exact logical duplicate comparison, conflict audit, and all-or-none insertion.';

commit;
